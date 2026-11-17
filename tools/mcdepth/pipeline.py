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
        self.rect = geometry.rectify(rig.r, rig.t)
        self.k_new = geometry.rectified_camera_matrix(
            rig.ref.k[0, 0], rig.ref.width, rig.aux.k[0, 0], rig.aux.width, self.out_w, self.out_h
        )
        self.correction = np.eye(3)
        self.baseline_sign = 1.0  # -1 if the images show the reported camera order is reversed
        self.self_calibrated = False  # True if depth is only relative after a coarse fit
        along = self.out_h if self.rect.vertical else self.out_w
        nd = max_disparity or along // 5
        self.num_disparities = max(16, int(np.ceil(nd / 16)) * 16)
        self._build_maps()

    # ---- rectification -------------------------------------------------
    def _build_maps(self):
        size = (self.out_w, self.out_h)
        rig = self.rig
        self.map_ref = cv2.initUndistortRectifyMap(rig.ref.k, rig.ref.dist, self.rect.r1, self.k_new, size, cv2.CV_16SC2)
        self.map_aux = cv2.initUndistortRectifyMap(
            rig.aux.k, rig.aux.dist, self.rect.r2, self.correction @ self.k_new, size, cv2.CV_16SC2
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
        2. Coarse (only if 1 fails): the calibration is badly off, so first fit
           scale, rotation and offset between the images, then run 1 again.
           Depth is then no longer metric (self.self_calibrated is set).

        Always starts from the nominal calibration, so repeated calls do not
        accumulate error."""
        self.correction = np.eye(3)
        self.baseline_sign = 1.0
        self.self_calibrated = False
        self._build_maps()
        gray_ref, gray_aux = _gray(img_ref), _gray(img_aux)
        info = self._fine(gray_ref, gray_aux)
        if not info["applied"]:
            coarse = self._coarse(info.pop("_p1"), info.pop("_p2"))
            if coarse is not None:
                self.correction = coarse
                self._build_maps()
                second = self._fine(gray_ref, gray_aux)
                if second["applied"]:
                    self.self_calibrated = True
                    info = second
                    info["coarse"] = True
                else:
                    self.correction = np.eye(3)
                    self._build_maps()
        info.pop("_p1", None)
        info.pop("_p2", None)
        return info

    def _usable(self, p1, p2, sign):
        """Matches that could be a stereo correspondence for a baseline pointing along `sign`."""
        ax, other = self.rect.axis, 1 - self.rect.axis
        d = (p1[:, ax] - p2[:, ax]) * sign
        lim = 0.05 * (self.out_w if self.rect.vertical else self.out_h)
        return (np.abs(p1[:, other] - p2[:, other]) < lim) & (d > -lim) & (d < self.num_disparities + lim)

    def _fine(self, gray_ref, gray_aux) -> dict:
        rect_ref, rect_aux = self.rectify(gray_ref, gray_aux)
        p1, p2 = match_features(rect_ref, rect_aux)
        info = {"matches": len(p1), "usable": 0, "inliers": 0, "applied": False, "_p1": p1, "_p2": p2}
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

    def _coarse(self, p1, p2):
        """Similarity transform taking aux points onto ref points, with the offset along
        the baseline chosen so that every disparity stays positive. None if unreliable."""
        if len(p1) < 20:
            return None
        m, inl = cv2.estimateAffinePartial2D(
            p2.astype(np.float32), p1.astype(np.float32), method=cv2.RANSAC, ransacReprojThreshold=3.0
        )
        if m is None or inl.sum() < 15:
            return None
        scale = float(np.hypot(m[0, 0], m[1, 0]))
        angle = float(np.degrees(np.arctan2(m[1, 0], m[0, 0])))
        if not 0.5 < scale < 2.0 or abs(angle) > 15:
            return None
        h = np.vstack([m, [0, 0, 1.0]])
        # the fit put the dominant surface at zero disparity; anything farther would be
        # negative. Shift so the farthest matched points sit just above zero.
        ax, other = self.rect.axis, 1 - self.rect.axis
        moved = p2 @ m[:, :2].T + m[:, 2]
        near_row = np.abs(moved[:, other] - p1[:, other]) < 3.0
        d = (p1[near_row, ax] - moved[near_row, ax]) * np.sign(self.rect.baseline)
        d = d[np.abs(d) < self.num_disparities]
        if len(d) < 15:
            return None
        shift = 2.0 - np.percentile(d, 2)
        h[ax, 2] -= shift * np.sign(self.rect.baseline)
        return h

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
