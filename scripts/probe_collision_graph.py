"""Print which robot body pairs can collide in the built Newton model (1 env, headless)."""
import argparse, itertools, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_headless_env import apply_overrides  # noqa: E402
from simtoolreal_newton.cfg import SimToolRealCfg  # noqa: E402
from simtoolreal_newton.launch import make_env  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="PATH=VALUE")
    parser.add_argument("--contact", action="store_true", help="Enable the fingertip contact sensor.")
    args = parser.parse_args()
    env_cfg = SimToolRealCfg()
    env_cfg.env.num_envs = 1
    env_cfg.env.play = True
    env_cfg.viewer.enable_viewer = False
    env_cfg.viewer.reference_ghost = False
    env_cfg.viewer.training_camera_enabled = False
    if args.contact:
        env_cfg.contact.enabled = True
    apply_overrides(env_cfg, args.overrides)
    env = make_env(env_cfg, num_envs=None, device="cuda:0", physics=None, visualizer=None)
    from isaaclab_newton.physics import NewtonManager
    from newton import ShapeFlags
    COLLIDE = ShapeFlags.COLLIDE_SHAPES
    model = NewtonManager.get_model()
    body_key = list(model.body_label)
    shape_body = model.shape_body.numpy()
    shape_key = list(model.shape_label)
    groups = model.shape_collision_group.numpy()
    flags = model.shape_flags.numpy() if hasattr(model, "shape_flags") else None
    arr = model.shape_collision_filter_pairs_array(); pairs = set((int(a), int(b)) for a, b in arr.tolist())
    robot = [i for i, k in enumerate(body_key) if "Robot" in k or "rl_dg" in k or "_link" in k]
    print("bodies:", len(body_key), "robot bodies:", len(robot), "shapes:", len(shape_key), "filter pairs:", len(pairs))
    shapes_of = defaultdict(list)
    for s, b in enumerate(shape_body):
        shapes_of[int(b)].append(s)
    def short(k):
        return k.split("/")[-1]
    hand = [b for b in robot if "rl_dg" in body_key[b] or "wrist_3" in body_key[b]]
    for b in hand:
        cols = [s for s in shapes_of[b] if flags is None or (int(flags[s]) & int(COLLIDE))]
        print("  body", short(body_key[b]), "shapes", len(shapes_of[b]), "collide-flag", len(cols), "groups", sorted(set(int(groups[s]) for s in shapes_of[b])))

    def unfiltered_pairs(candidates, label):
        print(label)
        found = []
        for a, b in itertools.combinations(candidates, 2):
            sa, sb = shapes_of[a], shapes_of[b]
            if not sa or not sb:
                continue
            n = sum(1 for x in sa for y in sb if (min(x, y), max(x, y)) not in pairs)
            if n:
                found.append((short(body_key[a]), short(body_key[b]), n))
        for a, b, n in sorted(found):
            print("   ", a, "<->", b, n, "shape pairs")
        print("total:", len(found))
        return found

    hand_pairs = unfiltered_pairs(hand, "hand body pairs NOT fully filtered (can collide):")
    arm = [b for b in robot if b not in hand]
    unfiltered_pairs(arm, "arm-only body pairs NOT fully filtered (can collide):")
    cross = []
    for a in arm:
        for b in hand:
            sa, sb = shapes_of[a], shapes_of[b]
            n = sum(1 for x in sa for y in sb if (min(x, y), max(x, y)) not in pairs)
            if n:
                cross.append((short(body_key[a]), short(body_key[b]), n))
    print("arm-hand body pairs NOT fully filtered:", len(cross))
    for a, b, n in sorted(cross):
        print("   ", a, "<->", b, n, "shape pairs")
    print("total hand body pairs able to collide:", len(hand_pairs))
    env.close()


if __name__ == "__main__":
    main()
