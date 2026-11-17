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
        """Estimate the residual row misalignment from feature matches and fold
        it into the aux rectification map.  Always starts from the nominal
        calibration, so calling it repeatedly does not accumulate error."""
        self.correction = np.eye(3)
        self._build_maps()
        rect_ref, rect_aux = self.rectify(_gray(img_ref), _gray(img_aux))
        p1, p2 = match_features(rect_ref, rect_aux)
        ax = self.rect.axis
        other = 1 - ax
        info = {"matches": len(p1), "inliers": 0, "applied": False}
        if len(p1):
            # keep only matches that could be a valid stereo correspondence
            d = (p1[:, ax] - p2[:, ax]) * np.sign(self.rect.baseline)
            lim = 0.05 * (self.out_w if self.rect.vertical else self.out_h)
            ok = (np.abs(p1[:, other] - p2[:, other]) < lim) & (d > -lim) & (d < self.num_disparities + lim)
            p1, p2 = p1[ok], p2[ok]
            info["before_px"] = float(np.median(np.abs(p1[:, other] - p2[:, other]))) if len(p1) else None
        h, inliers = geometry.fit_row_alignment(p1, p2, self.rect.vertical, (self.out_w / 2.0, self.out_h / 2.0))
        info["inliers"] = inliers
        if h is not None:
            self.correction = h
            self._build_maps()
            info["applied"] = True
        return info

    # ---- matching --------------------------------------------------------
    def disparity(self, rect_ref, rect_aux) -> np.ndarray:
        """Disparity (pixels, > 0) for every pixel of the rectified reference
        image, NaN where no match was found."""
        left, right = _gray(rect_ref), _gray(rect_aux)
        if self.rect.vertical:
            left, right = cv2.transpose(left), cv2.transpose(right)
        flipped = self.rect.baseline < 0  # reference camera is the right-hand one
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
