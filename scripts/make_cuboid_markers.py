#!/usr/bin/env python3
"""Printable AprilTag markers and matching board definitions for the scaled bars.

Three bars are used to test the policy's generalisation over object size: the
4x4x12 cm bar (scale 0.8), the 5x5x15 cm one the demonstrations were recorded
with (scale 1.0) and the 6x6x18 cm one (scale 1.2). Each carries 14 markers --
three on each of the four long faces, one on each square end -- laid out exactly
as the existing bar's board, and each bar gets its own block of ids so the
estimator can tell which one it is looking at and therefore what scale to report.

The 5x5x15 bar and its config are left exactly as they are; only the two new
bars are produced here, in their own family and their own config.

Geometry is emitted at the exact nominal size of the bar -- the faces are put
where a 40/50/60 mm bar really has them -- rather than measured, so the six
faces agree with one another by construction. (The board in use today does not:
its two cross-section dimensions come out 54.85 and 46.67 mm instead of 50, and
markers on one physical face disagree by up to 3.3 mm about where that face is.)

Run it with the robohand venv, which carries cv2 and matplotlib:

    /home/duplo/git/robohand-robohand2/.venv/bin/python \
        scripts/make_cuboid_markers.py --out deploy/markers
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Markers sit on the face with a tenth of the face width of white around them,
# the proportion the working bar uses (40 mm tag on a 50 mm face).
TAG_FRACTION_OF_FACE = 0.8
# The 5x5x15 bar in use keeps its tag16h5 markers untouched, so the new bars need
# a family of their own: tag16h5 holds 30 ids in total and that bar already spends
# 14 of them. One apriltag detector takes one family (verified -- the binding
# rejects "tag16h5 tag36h11"), but running a second detector over the same frame
# costs only ~7 ms at 1280x720 against a loop that takes 40-50 ms, so all three
# bars can be recognised at once. tag36h11 is chosen for its 587 ids: no integer
# can collide with the old bar's 0-13, which matters because boards are matched to
# markers by id alone. Its 8x8 grid is the finest of the candidates -- about 6 px
# per cell for the small bar's 32 mm tags at 0.65 m -- which detects cleanly in a
# rendered test; if it proves marginal on the real, blurred, grazing-angle image,
# tag25h9 (7x7) is the fallback and needs its ids offset instead.
DICTIONARY = "tag36h11"
OPENCV_DICTIONARY = "DICT_APRILTAG_36h11"

# (name, scale, long edge m, cross-section m, first id). The id blocks are spaced
# so that the bar is readable from any single id in a log.
BARS = (
    ("cuboid_4x4x12", 0.8, 0.120, 0.040, 100),
    ("cuboid_6x6x18", 1.2, 0.180, 0.060, 200),
)

# id offset -> (outward face normal, in-plane u, in-plane v). u x v = normal,
# which makes the corner order below wind the same way as the working board.
# u runs along the bar's long axis on the four long faces, so the three markers
# of a face spread along it, and v works out to n x x_hat -- which on the bench
# is one gesture: put that face up with +x to your right and the tag's top edge
# points away from you. Both end faces use v = +z, so each is simply upright when
# read from outside with +z up.
X, Y, Z = np.eye(3)
FACES = (
    (-Z, X, -Y), (-Z, X, -Y), (-Z, X, -Y),      # ids +0,+1,+2
    (Y, X, -Z), (Y, X, -Z), (Y, X, -Z),         # ids +3,+4,+5
    (Z, X, Y), (Z, X, Y), (Z, X, Y),            # ids +6,+7,+8
    (-Y, X, Z), (-Y, X, Z), (-Y, X, Z),         # ids +9,+10,+11
    (-X, -Y, Z),                                # id +12
    (X, Y, Z),                                  # id +13
)
ALONG = (-1, 0, 1) * 4 + (0, 0)                 # position along u, in units of L/3


def board_definition(long_m: float, width_m: float, first_id: int) -> dict:
    """The 14 markers of one bar, in its own frame, at exact nominal size."""
    half = 0.5 * TAG_FRACTION_OF_FACE * width_m
    markers = []
    for offset, ((normal, u, v), along) in enumerate(zip(FACES, ALONG)):
        centre = normal * (0.5 * (long_m if abs(normal[0]) > 0.5 else width_m))
        if abs(normal[0]) < 0.5:                # a long face: spread along x
            centre = centre + u * (along * long_m / 3.0)
        corners = [
            centre - half * u + half * v,
            centre + half * u + half * v,
            centre + half * u - half * v,
            centre - half * u - half * v,
        ]
        markers.append({"id": first_id + offset, "corners": [c.tolist() for c in corners]})
    return {"dictionary": DICTIONARY, "markers": markers}


def marker_cells(dictionary, marker_id: int) -> np.ndarray:
    """The marker as a (n+2, n+2) array of 0/1, one entry per printed cell."""
    import cv2

    side = dictionary.markerSize + 2
    image = cv2.aruco.generateImageMarker(dictionary, int(marker_id), side, borderBits=1)
    return (np.asarray(image) > 127).astype(int)


def draw_marker(axes, cells: np.ndarray, x_mm: float, y_mm: float, side_mm: float) -> None:
    """Draw one marker as vector rectangles, so the PDF prints crisp at any size."""
    from matplotlib.patches import Rectangle

    n = cells.shape[0]
    cell = side_mm / n
    axes.add_patch(Rectangle((x_mm, y_mm), side_mm, side_mm, facecolor="white", edgecolor="none"))
    for row in range(n):
        for column in range(n):
            if cells[row, column]:
                continue                        # white cell: leave the paper
            axes.add_patch(Rectangle(
                (x_mm + column * cell, y_mm + (n - 1 - row) * cell),
                cell, cell, facecolor="black", edgecolor="none", linewidth=0,
            ))


def sheets(name: str, scale: float, long_m: float, width_m: float, first_id: int, out: Path) -> Path:
    """One PDF per bar: four long strips, two end squares, at exact scale."""
    import matplotlib
    matplotlib.use("Agg")
    import cv2
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Rectangle

    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, OPENCV_DICTIONARY))
    long_mm, width_mm = 1000 * long_m, 1000 * width_m
    tag_mm = TAG_FRACTION_OF_FACE * width_mm
    pitch_mm = long_mm / 3.0

    # (label, ids, strip length, strip width, positions of the tags along it)
    pieces = [
        ("faccia -z", [first_id + 0, first_id + 1, first_id + 2], long_mm, width_mm, (-1, 0, 1)),
        ("faccia +y", [first_id + 3, first_id + 4, first_id + 5], long_mm, width_mm, (-1, 0, 1)),
        ("faccia +z", [first_id + 6, first_id + 7, first_id + 8], long_mm, width_mm, (-1, 0, 1)),
        ("faccia -y", [first_id + 9, first_id + 10, first_id + 11], long_mm, width_mm, (-1, 0, 1)),
        ("testa -x", [first_id + 12], width_mm, width_mm, (0,)),
        ("testa +x", [first_id + 13], width_mm, width_mm, (0,)),
    ]

    path = out / "{}_markers.pdf".format(name)
    margin, gap = 14.0, 9.0
    with PdfPages(path) as pdf:
        index = 0
        page = 1
        while index < len(pieces):
            figure = plt.figure(figsize=(210 / 25.4, 297 / 25.4))
            axes = figure.add_axes([0, 0, 1, 1])
            axes.set_xlim(0, 210); axes.set_ylim(0, 297)
            axes.set_aspect("equal"); axes.axis("off")
            axes.text(margin, 297 - 9, "{}  ({} x {} x {} mm, scala {})".format(
                name, int(width_mm), int(width_mm), int(long_mm), scale),
                fontsize=9, va="top")
            axes.text(margin, 297 - 15,
                      "{}  --  tag {:.0f} mm  --  stampare al 100%, NON adattare alla pagina".format(
                          DICTIONARY, tag_mm), fontsize=7, va="top", color="0.3")
            # A ruler to check the print scale before cutting anything.
            axes.add_patch(Rectangle((margin, 297 - 24), 100, 2.0, facecolor="black"))
            axes.text(margin + 102, 297 - 23, "100 mm esatti", fontsize=6, va="center", color="0.3")

            top = 297 - 34
            while index < len(pieces):
                label, ids, length, strip_width, positions = pieces[index]
                if top - strip_width < margin:
                    break
                axes.add_patch(Rectangle((margin, top - strip_width), length, strip_width,
                                         facecolor="none", edgecolor="0.75",
                                         linewidth=0.4, linestyle=(0, (4, 3))))
                for marker_id, along in zip(ids, positions):
                    cx = margin + 0.5 * length + along * (pitch_mm if len(ids) > 1 else 0.0)
                    cy = top - 0.5 * strip_width
                    draw_marker(axes, marker_cells(dictionary, marker_id),
                                cx - 0.5 * tag_mm, cy - 0.5 * tag_mm, tag_mm)
                    axes.text(cx, top - strip_width - 2.2, "id {}".format(marker_id),
                              fontsize=6, ha="center", va="top", color="0.35")
                axes.text(margin + length + 4, top - 0.5 * strip_width, label,
                          fontsize=7, va="center", color="0.35")
                top -= strip_width + gap + 4.0
                index += 1
            pdf.savefig(figure); plt.close(figure)
            page += 1
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="deploy/markers", help="where to write the PDFs and JSONs")
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    index = {}
    for name, scale, long_m, width_m, first_id in BARS:
        definition = board_definition(long_m, width_m, first_id)
        json_path = out / "{}.json".format(name)
        json_path.write_text(json.dumps(definition, indent=1) + "\n")
        pdf_path = sheets(name, scale, long_m, width_m, first_id, out)
        index[name] = {
            "scale": scale,
            "size_m": [long_m, width_m, width_m],
            "ids": [first_id, first_id + 13],
            "tag_mm": round(1000 * TAG_FRACTION_OF_FACE * width_m, 2),
            "board": json_path.name,
        }
        print("{:<16} ids {:>2}-{:<3} tag {:>5.1f} mm  ->  {}  {}".format(
            name, first_id, first_id + 13, 1000 * TAG_FRACTION_OF_FACE * width_m,
            json_path.name, pdf_path.name))
    (out / "bars.json").write_text(json.dumps(
        {"dictionary": DICTIONARY, "bars": index}, indent=1) + "\n")
    print("\nid -> scala:  " + ",  ".join(
        "{}-{} = {:g}".format(v["ids"][0], v["ids"][1], v["scale"]) for v in index.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
