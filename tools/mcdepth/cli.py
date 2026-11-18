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
from .pipeline import StereoDepth, colorize, fuse_disparities, warp_disparity, write_ply


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
    rig2 = sess.load_rig2(session)
    sd2 = StereoDepth(rig2, out_size=size) if rig2 is not None else None

    def load(row):
        return cv2.imread(str(session / row["ref_file"])), cv2.imread(str(session / row["aux_file"]))

    if not args.no_refine:
        middle = frames[len(frames) // 2]
        info = sd.refine(*load(middle))
        print("refine:", info)
        if sd2 is not None:
            print("refine third camera:", sd2.refine(load(middle)[0], cv2.imread(str(session / middle["aux2_file"]))))
    relative = sd.self_calibrated or not rig.metric
    print(
        f"baseline {abs(sd.rect.baseline) * 1000:.1f} mm ({'vertical' if sd.rect.vertical else 'horizontal'}), "
        f"focal {sd.k_new[0, 0]:.0f} px, depth is "
        + ("RELATIVE (geometry measured from the images, or baseline assumed)" if relative else "metric")
    )
    for row in frames:
        ref, aux = load(row)
        rect_ref, _, disp, depth = sd.compute(ref, aux)
        note = ""
        if sd2 is not None:
            _, _, disp2, _ = sd2.compute(ref, cv2.imread(str(session / row["aux2_file"])))
            disp, fusion = fuse_disparities(disp, warp_disparity(disp2, sd2, sd))
            depth = sd.depth_from_disparity(disp)
            note = f", third camera: {fusion}"
        stem = Path(row["ref_file"]).stem
        mm = np.where(np.isfinite(depth), np.clip(depth * 1000, 0, 65535), 0).astype(np.uint16)
        cv2.imwrite(str(out / f"{stem}_depth_mm.png"), mm)
        cv2.imwrite(str(out / f"{stem}_vis.jpg"), np.hstack([rect_ref, colorize(disp)]))
        if args.ply:
            write_ply(out / f"{stem}.ply", depth, rect_ref, sd.k_new)
        valid = np.isfinite(depth)
        med = float(np.median(depth[valid])) if valid.any() else float("nan")
        print(f"{stem}: {valid.mean() * 100:.0f}% valid, median depth {med:.2f} m{note}")
    print("wrote", out)


def demo(args):
    out = Path(args.dir)
    names = ["ref", "aux"] + (["aux2"] if args.three else [])
    for n in names:
        (out / n).mkdir(parents=True, exist_ok=True)
    if args.three:
        rig, rig2 = synth.phone_rig3()
    else:
        rig, rig2 = synth.phone_rig(tilt_deg=0.3), None
    sess.write_calibration(out, rig, {"device": "synthetic"}, rig2)
    with open(out / "frames.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "t_ref_ns", "t_aux_ns", "ref_file", "aux_file"] + (["t_aux2_ns", "aux2_file"] if rig2 else []))
        for i, near in enumerate((0.4, 0.6, 0.9)):
            t = i * 33_333_333
            ref, aux, _ = synth.render_pair(rig, near=near)
            cv2.imwrite(str(out / f"ref/{i:06d}.jpg"), ref)
            cv2.imwrite(str(out / f"aux/{i:06d}.jpg"), aux)
            row = [i, t, t, f"ref/{i:06d}.jpg", f"aux/{i:06d}.jpg"]
            if rig2 is not None:
                aux2, _ = synth.render(rig2.aux, rig2.r, rig2.t, near=near)
                cv2.imwrite(str(out / f"aux2/{i:06d}.jpg"), aux2)
                row += [t, f"aux2/{i:06d}.jpg"]
            w.writerow(row)
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
    d.add_argument("--three", action="store_true", help="include a third (telephoto) camera")
    d.set_defaults(func=demo)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
