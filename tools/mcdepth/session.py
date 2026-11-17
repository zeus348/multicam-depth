"""Read / write the recording format produced by the Android app.
See docs/session-format.md."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .pipeline import Camera, StereoRig


def _camera(d) -> Camera:
    fx, fy, cx, cy = d["intrinsics"]
    k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    return Camera(d["width"], d["height"], k, np.array(d.get("distortion", [0] * 5), float))


def load_rig(session: Path) -> StereoRig:
    c = json.loads((session / "calibration.json").read_text())
    s = c["stereo"]
    return StereoRig(
        ref=_camera(c["ref"]),
        aux=_camera(c["aux"]),
        r=np.array(s["R"], float).reshape(3, 3),
        t=np.array(s["T"], float),
        metric=s.get("metric", True),
    )


def load_frames(session: Path) -> list[dict]:
    with open(session / "frames.csv") as f:
        return list(csv.DictReader(f))


def write_calibration(session: Path, rig: StereoRig, extra: dict | None = None):
    def cam(c: Camera, cid: str):
        return {
            "id": cid,
            "width": c.width,
            "height": c.height,
            "intrinsics": [c.k[0, 0], c.k[1, 1], c.k[0, 2], c.k[1, 2]],
            "distortion": list(map(float, c.dist)),
        }

    doc = {
        "format": 1,
        "ref": cam(rig.ref, "0"),
        "aux": cam(rig.aux, "1"),
        "stereo": {"R": rig.r.flatten().tolist(), "T": rig.t.tolist(), "metric": rig.metric},
    }
    doc.update(extra or {})
    (session / "calibration.json").write_text(json.dumps(doc, indent=2))
