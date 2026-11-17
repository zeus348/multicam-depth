"""Pure stereo geometry. Mirrored 1:1 by the Kotlin file
android/app/src/main/java/dev/multicamdepth/calib/StereoGeometry.kt
so the same numbers can be checked on both sides.

Conventions (OpenCV style):
  * camera frame: x right, y down, z forward, in the image as delivered by the sensor
  * relative pose maps a point from camera 1 (reference) to camera 2 (aux):
        X2 = R @ X1 + T
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def quat_to_rot(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Android LENS_POSE_ROTATION quaternion (x, y, z, w) -> 3x3 matrix.

    The matrix takes a point in the device *sensor* coordinate system to the
    camera-aligned coordinate system.
    """
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ]
    )


def relative_pose(q1, t1, q2, t2):
    """Relative pose from two Android lens poses.

    Android defines, for each camera i:  p_cam_i = R_i @ (p_sensor - t_i)
    with R_i from LENS_POSE_ROTATION and t_i = LENS_POSE_TRANSLATION (metres).
    """
    r1 = quat_to_rot(*q1)
    r2 = quat_to_rot(*q2)
    t1 = np.asarray(t1, float)
    t2 = np.asarray(t2, float)
    r = r2 @ r1.T
    t = r2 @ (t1 - t2)
    return r, t


def scale_intrinsics(fx, fy, cx, cy, array_w, array_h, out_w, out_h):
    """Intrinsics given in active-array pixels -> pixels of an output stream.

    Streams are produced by centre-cropping the active array to the stream's
    aspect ratio and scaling, so the scale is the "cover" scale.
    """
    s = max(out_w / array_w, out_h / array_h)
    return (
        fx * s,
        fy * s,
        (cx - array_w / 2.0) * s + out_w / 2.0,
        (cy - array_h / 2.0) * s + out_h / 2.0,
    )


def fallback_focal_px(focal_mm, sensor_w_mm, array_w):
    """Focal length in active-array pixels when no calibration is published."""
    return focal_mm / sensor_w_mm * array_w


@dataclass
class Rectification:
    r1: np.ndarray  # camera 1 -> rectified frame
    r2: np.ndarray  # camera 2 -> rectified frame
    baseline: float  # signed: position of camera 2 along the baseline axis
    vertical: bool  # True: baseline along image y, epipolar lines are columns

    @property
    def axis(self) -> int:
        return 1 if self.vertical else 0


def rectify(r: np.ndarray, t: np.ndarray) -> Rectification:
    """Rotations that make both image planes coplanar and row (or column) aligned.

    After rectification with a shared camera matrix (focal f), a point at depth Z
    satisfies   u1 - u2 = f * baseline / Z   along the baseline axis.
    """
    r = np.asarray(r, float)
    t = np.asarray(t, float)
    c2 = -r.T @ t  # centre of camera 2 in camera-1 coordinates
    norm = float(np.linalg.norm(c2))
    if norm < 1e-9:
        raise ValueError("cameras are at the same position; no stereo baseline")
    vertical = abs(c2[1]) > abs(c2[0])
    k = 1 if vertical else 0
    sign = 1.0 if c2[k] >= 0 else -1.0
    e = c2 / norm * sign

    z_mean = np.array([0.0, 0.0, 1.0]) + r[2, :]  # r[2,:] = z axis of cam 2 in cam-1 coords
    z = z_mean - (z_mean @ e) * e
    z /= np.linalg.norm(z)
    if vertical:
        y = e
        x = np.cross(y, z)
    else:
        x = e
        y = np.cross(z, x)
    r_rect = np.stack([x, y, z])
    return Rectification(r1=r_rect, r2=r_rect @ r.T, baseline=sign * norm, vertical=vertical)


def rectified_camera_matrix(f1_px, w1, f2_px, w2, out_w, out_h):
    """Shared camera matrix of the rectified pair.

    Uses the longer focal length (narrower camera), because only the narrower
    field of view is seen by both cameras.
    """
    f = max(f1_px * out_w / w1, f2_px * out_w / w2)
    return np.array([[f, 0, out_w / 2.0], [0, f, out_h / 2.0], [0, 0, 1.0]])


def fit_row_alignment(p1, p2, vertical, center, iters=300, thresh=1.0, seed=7):
    """Residual-misalignment correction estimated from matched points.

    p1, p2: (N,2) matched pixel positions in the rectified images.
    Finds (a, b, c) with  v1 ~= a*u2 + b*v2 + c  where v is the coordinate that
    should be identical in both images (row for horizontal rigs, column for
    vertical ones) and returns the 3x3 homography to apply to image 2, plus the
    inlier count.  Returns (None, n) when the fit is not trustworthy.
    """
    p1 = np.asarray(p1, float)
    p2 = np.asarray(p2, float)
    n = len(p1)
    if n < 12:
        return None, n
    if vertical:  # swap so that index 1 is always the coordinate to align
        p1 = p1[:, ::-1]
        p2 = p2[:, ::-1]
        center = (center[1], center[0])
    cu, cv = center
    u = p2[:, 0] - cu
    v = p2[:, 1] - cv
    target = p1[:, 1] - cv
    design = np.stack([u, v, np.ones(n)], axis=1)

    rng = np.random.default_rng(seed)
    best = None
    best_count = 0
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        try:
            sol = np.linalg.solve(design[idx], target[idx])
        except np.linalg.LinAlgError:
            continue
        inl = np.abs(design @ sol - target) < thresh
        count = int(inl.sum())
        if count > best_count:
            best_count, best = count, inl
    if best is None or best_count < 12:
        return None, best_count
    for _ in range(2):  # refine on inliers
        sol, *_ = np.linalg.lstsq(design[best], target[best], rcond=None)
        best = np.abs(design @ sol - target) < thresh
        if best.sum() < 12:
            return None, int(best.sum())
    a, b, c = sol
    if abs(a) > 0.1 or abs(b - 1) > 0.1:
        return None, int(best.sum())
    c_abs = c + cv - a * cu - b * cv
    if vertical:
        h = np.array([[b, a, c_abs], [0, 1, 0], [0, 0, 1.0]])
    else:
        h = np.array([[1, 0, 0], [a, b, c_abs], [0, 0, 1.0]])
    return h, int(best.sum())
