"""python -m mcdepth.cli process <session> [--out DIR] [--ply] [--no-refine]
python -m mcdepth.cli demo <dir>        write a synthetic session to try the tool on
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from . import session as sess
from . import synth
from .pipeline import StereoDepth, colorize, write_ply


def process(args):
    session = Path(args.session)
    out = Path(args.out) if args.out else session / "depth"
    out.mkdir(parents=True, exist_ok=True)
    rig = sess.load_rig(session)
    frames = sess.load_frames(session)[:: args.every]
    if not frames:
        raise SystemExit("frames.csv is empty")
    size = (rig.ref.width // args.downscale, rig.ref.height // args.downscale)
    sd = StereoDepth(rig, out_size=size)

    def load(row):
        return cv2.imread(str(session / row["ref_file"])), cv2.imread(str(session / row["aux_file"]))

    if not args.no_refine:
        info = sd.refine(*load(frames[len(frames) // 2]))
        print("refine:", info)
    relative = sd.self_calibrated or not rig.metric
    print(
        f"baseline {abs(sd.rect.baseline) * 1000:.1f} mm ({'vertical' if sd.rect.vertical else 'horizontal'}), "
        f"focal {sd.k_new[0, 0]:.0f} px, depth is "
        + ("RELATIVE (geometry measured from the images, or baseline assumed)" if relative else "metric")
    )
    for row in frames:
        ref, aux = load(row)
        rect_ref, _, disp, depth = sd.compute(ref, aux)
        stem = Path(row["ref_file"]).stem
        mm = np.where(np.isfinite(depth), np.clip(depth * 1000, 0, 65535), 0).astype(np.uint16)
        cv2.imwrite(str(out / f"{stem}_depth_mm.png"), mm)
        cv2.imwrite(str(out / f"{stem}_vis.jpg"), np.hstack([rect_ref, colorize(disp, sd.num_disparities)]))
        if args.ply:
            write_ply(out / f"{stem}.ply", depth, rect_ref, sd.k_new)
        valid = np.isfinite(depth)
        med = float(np.median(depth[valid])) if valid.any() else float("nan")
        print(f"{stem}: {valid.mean() * 100:.0f}% valid, median depth {med:.2f} m")
    print("wrote", out)


def demo(args):
    out = Path(args.dir)
    (out / "ref").mkdir(parents=True, exist_ok=True)
    (out / "aux").mkdir(exist_ok=True)
    rig = synth.phone_rig(tilt_deg=0.3)
    sess.write_calibration(out, rig, {"device": "synthetic"})
    with open(out / "frames.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "t_ref_ns", "t_aux_ns", "ref_file", "aux_file"])
        for i, near in enumerate((0.4, 0.6, 0.9)):
            ref, aux, _ = synth.render_pair(rig, near=near)
            cv2.imwrite(str(out / f"ref/{i:06d}.jpg"), ref)
            cv2.imwrite(str(out / f"aux/{i:06d}.jpg"), aux)
            w.writerow([i, i * 33_333_333, i * 33_333_333, f"ref/{i:06d}.jpg", f"aux/{i:06d}.jpg"])
    print("wrote", out)


def main():
    p = argparse.ArgumentParser(prog="mcdepth")
    sub = p.add_subparsers(required=True)
    a = sub.add_parser("process", help="compute depth for a recorded session")
    a.add_argument("session")
    a.add_argument("--out")
    a.add_argument("--every", type=int, default=1, help="process every Nth frame pair")
    a.add_argument("--downscale", type=int, default=1)
    a.add_argument("--ply", action="store_true", help="also write coloured point clouds")
    a.add_argument("--no-refine", action="store_true", help="trust the recorded calibration as is")
    a.set_defaults(func=process)
    d = sub.add_parser("demo", help="write a synthetic session")
    d.add_argument("dir")
    d.set_defaults(func=demo)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
