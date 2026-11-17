"""Rectify -> (optionally refine) -> match -> depth.

The Android app runs the same steps in DepthEngine.kt.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import geometry


@dataclass
class Camera:
    width: int
    height: int
    k: np.ndarray  # 3x3, pixels of this camera's images
    dist: np.ndarray = field(default_factory=lambda: np.zeros(5))  # OpenCV order k1 k2 p1 p2 k3


@dataclass
class StereoRig:
    ref: Camera
    aux: Camera
    r: np.ndarray  # X_aux = r @ X_ref + t
    t: np.ndarray  # metres
    metric: bool = True  # False when the baseline was assumed, depth is then only relative


class StereoDepth:
    def __init__(self, rig: StereoRig, out_size: tuple[int, int] | None = None, max_disparity: int | None = None):
        self.rig = rig
        self.out_w, self.out_h = out_size or (rig.ref.width, rig.ref.height)
        self.k_new = geometry.rectified_camera_matrix(
            rig.ref.k[0, 0], rig.ref.width, rig.aux.k[0, 0], rig.aux.width, self.out_w, self.out_h
        )
        self.max_disparity = max_disparity
        self._reset()

    def _reset(self):
        """Back to the calibration as given."""
        self.correction = np.eye(3)
        self.baseline_sign = 1.0  # -1 if the images show the reported camera order is reversed
        self.self_calibrated = False  # True if the geometry was measured from the images: depth is relative
        self._set_geometry(self.rig.r, self.rig.t, self.rig.aux.k)

    def _set_geometry(self, r, t, aux_k):
        self.aux_k = aux_k
        self.rect = geometry.rectify(r, t)
        along = self.out_h if self.rect.vertical else self.out_w
        nd = self.max_disparity or along // 5
        self.num_disparities = max(16, int(np.ceil(nd / 16)) * 16)
        self._build_maps()

    # ---- rectification -------------------------------------------------
    def _build_maps(self):
        size = (self.out_w, self.out_h)
        rig = self.rig
        self.map_ref = cv2.initUndistortRectifyMap(rig.ref.k, rig.ref.dist, self.rect.r1, self.k_new, size, cv2.CV_16SC2)
        self.map_aux = cv2.initUndistortRectifyMap(
            self.aux_k, rig.aux.dist, self.rect.r2, self.correction @ self.k_new, size, cv2.CV_16SC2
        )

    def rectify(self, img_ref, img_aux):
        a = cv2.remap(img_ref, *self.map_ref, cv2.INTER_LINEAR)
        b = cv2.remap(img_aux, *self.map_aux, cv2.INTER_LINEAR)
        return a, b

    # ---- refinement from image content ----------------------------------
    def refine(self, img_ref, img_aux) -> dict:
        """Align the rectified pair using feature matches.

        1. Fine: fit the leftover row misalignment, assuming the calibration is
           nearly right. Also detects a baseline that points the wrong way.
        2. Self-calibration (only if 1 fails): the calibration is unusable, which is
           common on phones that publish placeholder values. Measure how the second
           image is scaled, rotated and shifted against the first, and in which
           direction parallax runs, rebuild the geometry from that, then run 1 again.
           Depth is then relative, not metric (self.self_calibrated is set).

        Always starts from the calibration as given, so repeated calls do not
        accumulate error."""
        self._reset()
        gray_ref, gray_aux = _gray(img_ref), _gray(img_aux)
        first = self._fine(gray_ref, gray_aux)
        # A wrong calibration can still line up one surface of the scene; a right one
        # lines up most of it.
        if first["applied"] and first["inliers"] >= 0.5 * first["usable"]:
            return first
        self._reset()
        detail = self._self_calibrate(gray_ref, gray_aux)
        if detail is not None:
            second = self._fine(gray_ref, gray_aux)
            if second["applied"] and second["inliers"] >= first["inliers"]:
                self.self_calibrated = True
                return {**second, **detail, "self_calibrated": True}
        self._reset()
        return self._fine(gray_ref, gray_aux) if first["applied"] else first

    def _self_calibrate(self, gray_ref, gray_aux):
        size = (self.out_w, self.out_h)
        rig = self.rig
        eye = np.eye(3)
        # both images at the shared focal length, otherwise untouched
        a = cv2.remap(gray_ref, *cv2.initUndistortRectifyMap(rig.ref.k, rig.ref.dist, eye, self.k_new, size, cv2.CV_16SC2), cv2.INTER_LINEAR)
        b = cv2.remap(gray_aux, *cv2.initUndistortRectifyMap(rig.aux.k, rig.aux.dist, eye, self.k_new, size, cv2.CV_16SC2), cv2.INTER_LINEAR)
        p1, p2 = match_features(a, b)
        if len(p1) < 20:
            return None
        m, inl = cv2.estimateAffinePartial2D(
            p2.astype(np.float32), p1.astype(np.float32), method=cv2.RANSAC, ransacReprojThreshold=3.0
        )
        if m is None or inl.sum() < 15:
            return None
        f, cx, cy = self.k_new[0, 0], self.k_new[0, 2], self.k_new[1, 2]
        # as a camera rotation plus a focal-length correction, which unlike the 2D fit
        # is also right towards the image edges
        r_h, scale = geometry.rotation_from_similarity(m, f, cx, cy)
        angle = float(np.degrees(np.arccos(np.clip((np.trace(r_h) - 1) / 2, -1, 1))))
        if not 0.5 < scale < 2.0 or angle > 20:
            return None
        k_inv = np.linalg.inv(self.k_new)
        rays1 = np.c_[p1, np.ones(len(p1))] @ k_inv.T
        rays2 = np.c_[p2, np.ones(len(p2))] @ k_inv.T * [scale, scale, 1.0]  # aux rays
        r_h, agree = geometry.refine_rotation(r_h, rays2, rays1[:, :2], inl.ravel() > 0, tol=2.0 / f)
        if agree < 15:
            return None
        in_ref = rays2 @ r_h.T
        moved = in_ref[:, :2] / in_ref[:, 2:] * f + [cx, cy]

        # direction of the baseline in the image: measured from parallax if the scene
        # has any, otherwise as reported. Its sign always comes from the report.
        c2 = -rig.r.T @ rig.t
        hint = c2[:2] / max(np.linalg.norm(c2[:2]), 1e-12)
        limit = max(self.out_w, self.out_h) / 5.0
        direction, votes = geometry.parallax_direction(p1 - moved, max_len=limit)
        measured = direction is not None
        if not measured:
            direction = hint
        elif direction @ hint < 0:
            direction = -direction
        baseline = float(np.linalg.norm(c2))
        r_rel = r_h.T
        aux_k = rig.aux.k.copy()
        aux_k[0, 0] /= scale
        aux_k[1, 1] /= scale
        self._set_geometry(r_rel, -baseline * r_rel @ np.array([direction[0], direction[1], 0.0]), aux_k)

        # the fit put the dominant surface at zero disparity, so anything farther would
        # be negative: shift until the farthest matched points sit just above zero
        def project(rot, rays):
            q = rays @ rot.T
            return q[:, :2] / q[:, 2:] * f + [cx, cy]

        ax, other = self.rect.axis, 1 - self.rect.axis
        r1 = project(self.rect.r1, rays1)
        r2 = project(self.rect.r2, rays2)
        sign = np.sign(self.rect.baseline)
        row = np.abs(r1[:, other] - r2[:, other]) < 3.0
        d = (r1[row, ax] - r2[row, ax]) * sign
        d = d[np.abs(d) < self.num_disparities]
        if len(d) < 15:
            self._reset()
            return None
        self.correction = np.eye(3)
        self.correction[ax, 2] = -(2.0 - np.percentile(d, 2)) * sign
        self._build_maps()
        return {
            "scale": scale,
            "rotation_deg": angle,
            "offset_px": [float(m[0, 2]), float(m[1, 2])],
            "baseline_direction": [float(direction[0]), float(direction[1])],
            "direction_measured": measured,
            "direction_votes": votes,
        }

    def _usable(self, p1, p2, sign):
        """Matches that could be a stereo correspondence for a baseline pointing along `sign`."""
        ax, other = self.rect.axis, 1 - self.rect.axis
        d = (p1[:, ax] - p2[:, ax]) * sign
        lim = 0.05 * (self.out_w if self.rect.vertical else self.out_h)
        return (np.abs(p1[:, other] - p2[:, other]) < lim) & (d > -lim) & (d < self.num_disparities + lim)

    def _fine(self, gray_ref, gray_aux) -> dict:
        rect_ref, rect_aux = self.rectify(gray_ref, gray_aux)
        p1, p2 = match_features(rect_ref, rect_aux)
        info = {"matches": len(p1), "usable": 0, "inliers": 0, "applied": False}
        if not len(p1):
            return info
        nominal = np.sign(self.rect.baseline)
        sign = 1.0
        # Nearly all true disparities have the same sign. If that sign is the opposite of
        # what the calibration says, the reported camera order is the wrong way round.
        ax, other = self.rect.axis, 1 - self.rect.axis
        lim = 0.05 * (self.out_w if self.rect.vertical else self.out_h)
        d = (p1[:, ax] - p2[:, ax]) * nominal
        rows = (np.abs(p1[:, other] - p2[:, other]) < lim) & (np.abs(d) < self.num_disparities + lim)
        if rows.sum() >= 30 and np.median(d[rows]) < -2.0:
            sign = -1.0
        ok = self._usable(p1, p2, nominal * sign)
        info["usable"] = int(ok.sum())
        h, inliers = geometry.fit_row_alignment(p1[ok], p2[ok], self.rect.vertical, (self.out_w / 2.0, self.out_h / 2.0))
        info["inliers"] = inliers
        if h is not None:
            self.correction = h @ self.correction
            self.baseline_sign = sign
            self._build_maps()
            info["applied"] = True
            info["baseline_flipped"] = sign < 0
        return info

    # ---- matching --------------------------------------------------------
    def disparity(self, rect_ref, rect_aux) -> np.ndarray:
        """Disparity (pixels, > 0) for every pixel of the rectified reference
        image, NaN where no match was found."""
        left, right = _gray(rect_ref), _gray(rect_aux)
        if self.rect.vertical:
            left, right = cv2.transpose(left), cv2.transpose(right)
        flipped = self.rect.baseline * self.baseline_sign < 0  # reference camera is the right-hand one
        if flipped:
            left, right = cv2.flip(left, 1), cv2.flip(right, 1)
        block = 5
        sgbm = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=self.num_disparities,
            blockSize=block,
            P1=8 * block * block,
            P2=32 * block * block,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        disp = sgbm.compute(left, right).astype(np.float32) / 16.0
        if flipped:
            disp = cv2.flip(disp, 1)
        if self.rect.vertical:
            disp = cv2.transpose(disp)
        disp[disp <= 0] = np.nan
        return disp

    def depth_from_disparity(self, disp: np.ndarray) -> np.ndarray:
        """Depth along the optical axis, metres (if rig.metric)."""
        return self.k_new[0, 0] * abs(self.rect.baseline) / disp

    def compute(self, img_ref, img_aux):
        rect_ref, rect_aux = self.rectify(img_ref, img_aux)
        disp = self.disparity(rect_ref, rect_aux)
        return rect_ref, rect_aux, disp, self.depth_from_disparity(disp)


def _gray(img):
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def match_features(a, b, max_features=2000):
    """ORB + cross-checked brute force matching. Returns two (N,2) arrays."""
    orb = cv2.ORB_create(nfeatures=max_features)
    ka, da = orb.detectAndCompute(a, None)
    kb, db = orb.detectAndCompute(b, None)
    if da is None or db is None:
        return np.zeros((0, 2)), np.zeros((0, 2))
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
    matches = [m for m in matches if m.distance < 64]
    p1 = np.array([ka[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    p2 = np.array([kb[m.trainIdx].pt for m in matches]).reshape(-1, 2)
    return p1, p2


def colorize(disp: np.ndarray, max_disparity: float) -> np.ndarray:
    """Near = warm, far = cool, no data = black."""
    valid = np.isfinite(disp)
    norm = np.zeros(disp.shape, np.uint8)
    norm[valid] = np.clip(disp[valid] / max_disparity * 255, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    return color


def write_ply(path, depth, color_bgr, k):
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w]
    ok = np.isfinite(depth)
    z = depth[ok]
    x = (u[ok] - k[0, 2]) / k[0, 0] * z
    y = (v[ok] - k[1, 2]) / k[1, 1] * z
    if color_bgr.ndim == 2:
        color_bgr = cv2.cvtColor(color_bgr, cv2.COLOR_GRAY2BGR)
    c = color_bgr[ok]
    with open(path, "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(z)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for i in range(len(z)):
            f.write(f"{x[i]:.4f} {y[i]:.4f} {z[i]:.4f} {c[i, 2]} {c[i, 1]} {c[i, 0]}\n")
