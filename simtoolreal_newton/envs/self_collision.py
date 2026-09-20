"""Which robot body pairs the hand is allowed to collide with, and which to filter.

The Newton USD importer filters *every* body pair of an articulation whose
``newton:selfCollisionEnabled`` is false, which is what the port has shipped so
far: the hand passes through itself. Turning the articulation flag on would
switch on all 351 robot body pairs at once, most of which are permanently
interpenetrating convex hulls (the collapsed palm against its finger bases, a
finger's own adjacent links). This module names the small subset worth paying
for -- by default the 96 cross-finger pairs of the four long fingers -- so the
caller can author ``physics:filteredPairs`` for the complement.

Pure: no simulation, no USD. :mod:`tests.test_self_collision` covers it.
"""

from itertools import combinations
from typing import Iterable

# The body the collapsed DG5F mount/base/palm shapes ride on.
PALM_BODY_NAME = "wrist_3_link"
# Finger index 1 is the thumb; links run proximal (1) to distal (4, which also
# carries the merged fixed tip).
FINGER_INDICES = (1, 2, 3, 4, 5)
LINK_INDICES = (1, 2, 3, 4)


def finger_body_name(finger: int, link: int) -> str:
    return "rl_dg_{}_{}".format(int(finger), int(link))


def finger_body_names(fingers: Iterable[int] = FINGER_INDICES) -> tuple:
    """Body names of the given fingers, proximal to distal."""
    return tuple(finger_body_name(f, l) for f in fingers for l in LINK_INDICES)


def _settings(asset_cfg):
    fingers = [int(f) for f in getattr(asset_cfg, "self_collision_fingers", ()) or ()]
    fingers = sorted(set(f for f in fingers if f in FINGER_INDICES))
    return (
        fingers,
        bool(getattr(asset_cfg, "self_collision_adjacent_fingers_only", False)),
        bool(getattr(asset_cfg, "self_collision_with_palm", False)),
        bool(getattr(asset_cfg, "self_collision_same_finger", False)),
    )


def extra_pairs(asset_cfg) -> set:
    """``asset.self_collision_extra_pairs``: named body pairs opened on top of the rules.

    Each entry is a two-element list of body names, e.g.
    ``[["rl_dg_4_4", "wrist_3_link"]]`` lets the ring finger's distal phalanx
    (with its merged tip) touch the palm without opening every palm pair.
    """
    pairs = set()
    for entry in list(getattr(asset_cfg, "self_collision_extra_pairs", []) or []):
        if len(entry) != 2 or entry[0] == entry[1]:
            raise ValueError("self_collision_extra_pairs entries must name two different bodies: {!r}".format(entry))
        pairs.add(frozenset((str(entry[0]), str(entry[1]))))
    return pairs


def allowed_body_pairs(asset_cfg, body_names: Iterable[str]) -> set:
    """``{frozenset({body_a, body_b})}`` allowed to collide, both bodies present.

    Empty whenever the master switch ``asset.self_collision`` is off, so the
    caller filters everything exactly as the importer used to.
    """
    present = set(body_names)
    if not bool(getattr(asset_cfg, "self_collision", False)):
        return set()
    fingers, adjacent_only, with_palm, same_finger = _settings(asset_cfg)
    allowed = set()

    def add(a, b):
        if a != b and a in present and b in present:
            allowed.add(frozenset((a, b)))

    for fa, fb in combinations(fingers, 2):
        if adjacent_only and abs(fa - fb) != 1:
            continue
        for la in LINK_INDICES:
            for lb in LINK_INDICES:
                add(finger_body_name(fa, la), finger_body_name(fb, lb))
    if with_palm:
        for f in fingers:
            for l in LINK_INDICES:
                add(PALM_BODY_NAME, finger_body_name(f, l))
    if same_finger:
        for f in fingers:
            for la, lb in combinations(LINK_INDICES, 2):
                # Adjacent links share a joint and always overlap; only the
                # links a curl can fold onto each other are worth a contact.
                if lb - la >= 2:
                    add(finger_body_name(f, la), finger_body_name(f, lb))
    for pair in extra_pairs(asset_cfg):
        a, b = tuple(pair)
        if a not in present or b not in present:
            raise ValueError("self_collision_extra_pairs names an unknown body: {!r}".format(sorted(pair)))
        add(a, b)
    return allowed


def filtered_body_pairs(asset_cfg, body_names: Iterable[str]) -> set:
    """Every robot body pair that must carry a ``physics:filteredPairs`` entry."""
    names = list(body_names)
    allowed = allowed_body_pairs(asset_cfg, names)
    return set(frozenset(pair) for pair in combinations(sorted(set(names)), 2)) - allowed
