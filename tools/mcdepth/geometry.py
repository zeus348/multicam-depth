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


def orientation_rotation(sensor_orientation: int, front_facing: bool = False) -> np.ndarray:
    """Device ("sensor") coordinates -> camera coordinates, derived from
    SENSOR_ORIENTATION alone. Used when a phone publishes a placeholder
    LENS_POSE_ROTATION that does not match how its images are oriented.

    Device axes: x right, y up, z out of the screen. Rotating the image
    clockwise by sensor_orientation makes it upright.
    """
    a = np.deg2rad(sensor_orientation)
    right = np.array([-1.0 if front_facing else 1.0, 0.0, 0.0])  # scene-right in device axes
    down = np.array([0.0, -1.0, 0.0])
    forward = np.array([0.0, 0.0, 1.0 if front_facing else -1.0])
    x = np.cos(a) * right + np.sin(a) * down
    y = -np.sin(a) * right + np.cos(a) * down
    return np.stack([x, y, forward])


def parallax_direction(residuals, min_len=4.0, max_len=200.0, tol=1.5, max_candidates=200):
    """Direction along which matched points are displaced by depth.

    After the two images are aligned on the dominant surface, every remaining
    displacement between matched points is parallax, which runs along the
    baseline. residuals: (N,2) displacement vectors. Returns (unit vector, inliers)
    or (None, inliers) when there is no clear common direction.
    """
    r = np.asarray(residuals, float).reshape(-1, 2)
    length = np.hypot(r[:, 0], r[:, 1])
    r = r[(length > min_len) & (length < max_len)]
    n = len(r)
    if n < 12:
        return None, n
    step = max(1, n // max_candidates)
    best, best_count = None, 0
    for cand in r[::step]:
        u = cand / np.hypot(cand[0], cand[1])
        inl = np.abs(r[:, 0] * u[1] - r[:, 1] * u[0]) < tol
        if inl.sum() > best_count:
            best_count, best = int(inl.sum()), inl
    if best_count < 12 or best_count < 0.5 * n:
        return None, best_count
    pts = r[best]
    scatter = pts.T @ pts
    w, v = np.linalg.eigh(scatter)
    return v[:, 1], best_count


def rotation_from_similarity(m, f, cx, cy):
    """Camera rotation that explains a measured image-to-image similarity.

    m: 2x3 similarity taking aux pixels onto ref pixels (both at focal f,
    principal point cx, cy). Returns (R, scale): R rotates aux rays into the
    ref frame, so  K R diag(scale, scale, 1) K^-1  equals m near the image
    centre but, unlike m, is also right towards the edges.
    """
    m = np.asarray(m, float)
    scale = float(np.hypot(m[0, 0], m[1, 0]))
    roll = np.arctan2(m[1, 0], m[0, 0])
    qx = m[0, 0] * cx + m[0, 1] * cy + m[0, 2]
    qy = m[1, 0] * cx + m[1, 1] * cy + m[1, 2]
    v = np.array([(qx - cx) / f, (qy - cy) / f, 1.0])
    v /= np.linalg.norm(v)
    axis = np.array([-v[1], v[0], 0.0])  # z cross v
    sin = float(np.linalg.norm(axis))
    point = np.eye(3)
    if sin > 1e-12:
        k = axis / sin
        kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        point = np.eye(3) + sin * kx + (1 - v[2]) * (kx @ kx)
    c, s_ = np.cos(roll), np.sin(roll)
    rz = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1.0]])
    return point @ rz, scale


def rodrigues(w) -> np.ndarray:
    """Rotation vector (axis * angle, radians) -> rotation matrix."""
    w = np.asarray(w, float)
    angle = float(np.linalg.norm(w))
    if angle < 1e-12:
        return np.eye(3)
    k = w / angle
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * kx + (1 - np.cos(angle)) * (kx @ kx)


def refine_rotation(r, rays_aux, target, mask, tol, iters=3):
    """Least-squares polish of a rotation that takes aux rays onto ref points.

    rays_aux: (N,3) rays in the aux frame. target: (N,2) normalised ref image
    coordinates of the same points. mask: which pairs to trust in the first
    round; later rounds use every pair that agrees within tol.
    Returns (rotation, number of agreeing pairs).
    """
    r = np.asarray(r, float)
    mask = np.asarray(mask, bool)
    for i in range(iters):
        q = rays_aux @ r.T
        xy = q[:, :2] / q[:, 2:]
        if i > 0:
            mask = np.hypot(*(target - xy).T) < tol
        if mask.sum() < 12:
            break
        x, y = xy[mask, 0], xy[mask, 1]
        # effect of a small extra rotation w on the projected point
        jx = np.stack([-x * y, 1 + x * x, -y], axis=1)
        jy = np.stack([-(1 + y * y), x * y, x], axis=1)
        d = (target - xy)[mask]
        w, *_ = np.linalg.lstsq(np.vstack([jx, jy]), np.r_[d[:, 0], d[:, 1]], rcond=None)
        r = rodrigues(w) @ r
    q = rays_aux @ r.T
    agree = np.hypot(*(target - q[:, :2] / q[:, 2:]).T) < tol
    return r, int(agree.sum())


def fit_line(x, y, tol=0.75, iters=200, seed=3):
    """Robust y = a*x + b. Returns (a, b, inlier count), or (None, None, n) if there
    are too few points or x does not vary enough to pin down the slope."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    if n < 50:
        return None, None, n
    lo, hi = np.percentile(x, [5, 95])
    if hi - lo < 1.0:
        return None, None, n
    rng = np.random.default_rng(seed)
    best, best_count = None, 0
    for _ in range(iters):
        i, j = rng.integers(0, n, 2)
        if abs(x[i] - x[j]) < 0.5 * (hi - lo):
            continue
        a = (y[i] - y[j]) / (x[i] - x[j])
        b = y[i] - a * x[i]
        inl = np.abs(a * x + b - y) < tol
        if inl.sum() > best_count:
            best_count, best = int(inl.sum()), inl
    if best is None or best_count < 50:
        return None, None, best_count
    a, b = np.polyfit(x[best], y[best], 1)
    return float(a), float(b), int((np.abs(a * x + b - y) < tol).sum())


def integrate_gyro(times_ns, rates, t0_ns, t1_ns) -> np.ndarray:
    """Rotation of the device between two instants from gyroscope samples.

    times_ns: (N,) sample times, rates: (N,3) rad/s in the device frame.
    Returns R with  p_at_t0 = R @ p_at_t1  for a direction p fixed to the device.
    """
    times_ns = np.asarray(times_ns)
    rates = np.asarray(rates, float)
    r = np.eye(3)
    for i in range(len(times_ns)):
        start = max(t0_ns, times_ns[i - 1] if i else times_ns[i])
        end = min(t1_ns, times_ns[i])
        if i == len(times_ns) - 1 and times_ns[i] < t1_ns:
            end = t1_ns  # hold the last sample to the end of the interval
        if end > start:
            r = r @ rodrigues(rates[i] * (end - start) * 1e-9)
    return r
