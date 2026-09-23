#!/usr/bin/env python3
"""An assembly picture for the scaled bars: the cuboid unfolded, markers in place.

Sticking the markers is the only part of the board that is not already exact --
the JSON says where each corner must end up, so a tag rotated by 90 degrees is
read perfectly and yields a wrong pose. This draws the bar's net with the real
marker images already turned the right way, and checks itself: every marker it
draws is folded back into 3D and compared against the board definition, so the
picture cannot disagree with the file it is meant to explain.

    /home/duplo/git/robohand-robohand2/.venv/bin/python scripts/make_cuboid_net.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

X, Y, Z = np.eye(3)

# The four long faces top to bottom, in the order in which they wrap: the bottom
# edge of one is the top edge of the next. Each is drawn with its own tag upright,
# which is possible exactly because v = n x x_hat holds for all four.
LONG_FACES = ((Z, Y), (-Y, Z), (-Z, -Y), (Y, -Z))       # (normal, v = paper up)
# The ends, unfolded about their shared edge with the +z strip. Rotating a face
# into the paper turns its tag: the +x end lands with its right edge pointing up
# the page, the -x end with its right edge pointing down it.
END_ROTATION_DEG = {"+x": 90, "-x": -90}


def faces_of(board: dict):
    """Group a board's markers by face, with each marker's u and v in 3D."""
    out = {}
    for marker in board["markers"]:
        corners = np.asarray(marker["corners"], dtype=float)
        u = corners[1] - corners[0]; u /= np.linalg.norm(u)
        v = corners[0] - corners[3]; v /= np.linalg.norm(v)
        normal = np.cross(u, v)
        axis = int(np.argmax(np.abs(normal)))
        name = "{}{}".format("+" if normal[axis] > 0 else "-", "xyz"[axis])
        out.setdefault(name, []).append({
            "id": marker["id"], "centre": corners.mean(0), "u": u, "v": v,
            "side": float(np.linalg.norm(corners[1] - corners[0])),
        })
    for markers in out.values():
        markers.sort(key=lambda m: m["centre"][0])
    return out


def draw(board: dict, name: str, scale: float, out: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import cv2
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, FancyArrow

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    faces = faces_of(board)
    long_mm = 1000 * 2 * max(abs(m["centre"][0]) for m in faces["+x"] + faces["-x"])
    width_mm = 1000 * 2 * abs(faces["+z"][0]["centre"][2])
    tag_mm = 1000 * faces["+z"][0]["side"]

    total_w = long_mm + 2 * width_mm + 80
    total_h = 4 * width_mm + 122
    figure = plt.figure(figsize=(total_w / 25.4, total_h / 25.4), dpi=110)
    axes = figure.add_axes([0, 0, 1, 1])
    axes.set_xlim(0, total_w); axes.set_ylim(0, total_h)
    axes.set_aspect("equal"); axes.axis("off")
    axes.add_patch(Rectangle((0, 0), total_w, total_h, facecolor="white"))

    left = width_mm + 60
    top = total_h - 56
    checks = []

    def put(marker, ox, oy, rotation_deg, paper_right_3d, paper_up_3d):
        cells = (np.asarray(cv2.aruco.generateImageMarker(
            dictionary, int(marker["id"]), dictionary.markerSize + 2, borderBits=1)) > 127).astype(int)
        n = cells.shape[0]
        cell = tag_mm / n
        theta = np.radians(rotation_deg)
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        for row in range(n):
            for column in range(n):
                if cells[row, column]:
                    continue
                local = np.array([column * cell - tag_mm / 2, (n - 1 - row) * cell - tag_mm / 2])
                p = rot @ local
                axes.add_patch(Rectangle((ox + p[0], oy + p[1]), cell, cell,
                                         facecolor="black", edgecolor="none", linewidth=0))
        # fold the drawn orientation back into 3D and compare with the board
        right_paper = rot @ np.array([1.0, 0.0])
        up_paper = rot @ np.array([0.0, 1.0])
        u3 = right_paper[0] * paper_right_3d + right_paper[1] * paper_up_3d
        v3 = up_paper[0] * paper_right_3d + up_paper[1] * paper_up_3d
        checks.append((marker["id"], float(np.dot(u3, marker["u"])), float(np.dot(v3, marker["v"]))))

    for normal, up in LONG_FACES:
        axis = int(np.argmax(np.abs(normal)))
        face = "{}{}".format("+" if normal[axis] > 0 else "-", "xyz"[axis])
        bottom = top - width_mm
        axes.add_patch(Rectangle((left, bottom), long_mm, width_mm,
                                 facecolor="none", edgecolor="0.55", linewidth=1.0))
        for marker in faces[face]:
            cx = left + long_mm / 2 + 1000 * marker["centre"][0]
            put(marker, cx, bottom + width_mm / 2, 0, X, up)
        axes.text(left - width_mm - 6, bottom + width_mm / 2,
                  "faccia {}\n{}".format(face, "  ".join(str(m["id"]) for m in faces[face])),
                  fontsize=8, ha="right", va="center", color="0.25", linespacing=1.5)
        if face == "+z":                                  # the ends hang off this strip
            for end, ox in (("+x", left + long_mm + width_mm / 2), ("-x", left - width_mm / 2)):
                axes.add_patch(Rectangle((ox - width_mm / 2, bottom), width_mm, width_mm,
                                         facecolor="none", edgecolor="0.55", linewidth=1.0))
                # An end face is drawn folded into the +z plane, so the paper
                # axes correspond to different 3D directions than on the strips.
                put(faces[end][0], ox, bottom + width_mm / 2, END_ROTATION_DEG[end],
                    (-1.0 if end == "+x" else 1.0) * Z, up)
                axes.text(ox, bottom + width_mm + 3.0,
                          "testa {}\nid {}".format(end, faces[end][0]["id"]),
                          fontsize=8, ha="center", va="bottom", color="0.25", linespacing=1.4)
        top = bottom
    axes.add_patch(FancyArrow(left + long_mm * 0.32, total_h - 26, long_mm * 0.36, 0,
                              width=0.8, head_width=5, head_length=8, color="0.15"))
    axes.text(left + long_mm * 0.5, total_h - 19, "+x   asse lungo", fontsize=10,
              ha="center", color="0.15")
    axes.text(12, total_h - 10, "{}   -   {:.0f} x {:.0f} x {:.0f} mm   -   scala {:g}   -   "
              "tag {:.0f} mm".format(name, width_mm, width_mm, long_mm, scale, tag_mm),
              fontsize=11, va="top", color="0.1", weight="bold")
    axes.text(12, 52,
              "Questo e' il cuboide APERTO. Le quattro strisce si avvolgono nell'ordine in cui "
              "sono disegnate, dall'alto verso il basso:\n"
              "il bordo inferiore di una striscia e' il bordo superiore della successiva. "
              "Le due teste si ripiegano ai lati della faccia +z.\n"
              "Regola di controllo, vale per tutte e quattro le facce lunghe: metti quella faccia "
              "in alto con +x alla tua destra, e il lato ALTO del tag punta lontano da te.",
              fontsize=9, va="top", color="0.3", linespacing=1.9)

    path = out / "{}_net.png".format(name)
    figure.savefig(path, dpi=110, facecolor="white"); plt.close(figure)
    bad = [c for c in checks if c[1] < 0.999 or c[2] < 0.999]
    print("{:<15} {:>2} marker disegnati, ripiegatura verificata: {}".format(
        name, len(checks), "TUTTI OK" if not bad else "ERRORI su {}".format([b[0] for b in bad])))
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="deploy/markers")
    args = parser.parse_args()
    out = Path(args.out)
    index = json.loads((out / "bars.json").read_text())
    for name, entry in index["bars"].items():
        board = json.loads((out / entry["board"]).read_text())
        print("  ->", draw(board, name, entry["scale"], out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
