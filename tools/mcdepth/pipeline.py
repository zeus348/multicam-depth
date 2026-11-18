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
        # the app uses the faster 3-direction matcher; offline the fuller one streaks less
        self.sgbm_mode = cv2.STEREO_SGBM_MODE_SGBM
        # re-measure the row misalignment for every frame; see geometry.row_shift_plane
        self.track_rows = True
        self.last_tracking = None
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
        nd = self.max_disparity or along // 8
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
        # which rectified pixels actually come from inside each camera's image
        self.valid_ref = self._coverage(rig.ref, self.map_ref)
        self.valid_aux = self._coverage(rig.aux, self.map_aux)

    @staticmethod
    def _coverage(cam, maps):
        full = np.full((cam.height, cam.width), 255, np.uint8)
        mask = cv2.remap(full, *maps, cv2.INTER_NEAREST)
        return cv2.erode(mask, np.ones((5, 5), np.uint8))

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
        # bring everything into the form the matcher expects: horizontal baseline,
        # reference camera on the left
        mats = [_gray(rect_ref), _gray(rect_aux), self.valid_ref, self.valid_aux]
        if self.rect.vertical:
            mats = [cv2.transpose(m) for m in mats]
        flipped = self.rect.baseline * self.baseline_sign < 0  # reference camera is the right-hand one
        if flipped:
            mats = [cv2.flip(m, 1) for m in mats]
        left, right, valid_left, valid_right = mats
        self.last_tracking = None
        if self.track_rows:
            right = self._track_rows(left, right)
        nd = self.num_disparities
        block = 5
        sgbm = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=nd,
            blockSize=block,
            P1=8 * block * block,
            P2=32 * block * block,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=2,
            mode=self.sgbm_mode,
        )

        # The matcher cannot fill the first `nd` columns; padding moves that blind strip
        # off the image.
        def pad(m):
            return cv2.copyMakeBorder(m, 0, 0, nd, 0, cv2.BORDER_CONSTANT, value=0)

        disp = sgbm.compute(pad(left), pad(right))[:, nd:].astype(np.float32) / 16.0
        # drop pixels outside the reference image and matches that land outside the second one
        h, w = disp.shape
        xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        landed = cv2.remap(valid_right, xs - np.maximum(disp, 0), ys, cv2.INTER_NEAREST)
        disp[(landed == 0) | (valid_left == 0)] = -1
        disp = fill_streaks(despeckle(disp))
        if flipped:
            disp = cv2.flip(disp, 1)
        if self.rect.vertical:
            disp = cv2.transpose(disp)
        disp[disp <= 0] = np.nan
        return disp

    def _track_rows(self, left, right, bands=3):
        """Shift the right image so its rows agree with the left for this frame."""
        h, w = left.shape

        def profiles(img):
            return [img[:, b * w // bands:(b + 1) * w // bands].mean(axis=1) for b in range(bands)]

        max_shift = max(4, h // 40)
        a, b, c, blocks = geometry.row_shift_plane(profiles(left), profiles(right), w, max_shift)
        self.last_tracking = {"shift_px": a, "blocks": blocks}
        if blocks == 0 or (abs(a) < 0.15 and abs(b) * w < 0.3 and abs(c) * h < 0.3):
            return right
        m = np.array([[1.0, 0.0, 0.0], [b, 1.0 + c, a]])
        return cv2.warpAffine(right, m, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)

    def depth_from_disparity(self, disp: np.ndarray) -> np.ndarray:
        """Depth along the optical axis, metres (if rig.metric)."""
        return self.k_new[0, 0] * abs(self.rect.baseline) / disp

    def compute(self, img_ref, img_aux):
        rect_ref, rect_aux = self.rectify(img_ref, img_aux)
        disp = self.disparity(rect_ref, rect_aux)
        return rect_ref, rect_aux, disp, self.depth_from_disparity(disp)


def despeckle(disp: np.ndarray) -> np.ndarray:
    """5x5 median over valid disparities (invalid = anything <= 0); removes thin streaks
    and isolated wrong matches while keeping edges."""
    as_u16 = np.clip(disp * 16, 0, 65535).astype(np.uint16)
    out = cv2.medianBlur(as_u16, 5).astype(np.float32) / 16.0
    out[(out <= 0) | (disp <= 0)] = -1
    return out


def fill_streaks(disp: np.ndarray, reach: int = 5) -> np.ndarray:
    """Close thin gaps that run along the baseline. The matcher leaves such dashes where
    its left-right check fails on a single row; a gap only a few rows tall with valid
    depth on both sides is filled from those neighbours. `disp` is in matcher form
    (baseline horizontal), invalid = anything <= 0."""
    kernel = np.ones((reach, 1), np.uint8)
    closed = cv2.morphologyEx(disp, cv2.MORPH_CLOSE, kernel)
    holes = (disp <= 0) & (closed > 0)
    out = disp.copy()
    out[holes] = closed[holes]
    return out


def warp_disparity(disp, src: "StereoDepth", dst: "StereoDepth") -> np.ndarray:
    """Resample a disparity map from src's rectified reference view into dst's. Both must
    share the same reference camera, so the two views differ only by a rotation and focal
    length. Values are not rescaled; see fuse_disparities."""
    h = src.k_new @ src.rect.r1 @ dst.rect.r1.T @ np.linalg.inv(dst.k_new)  # dst pixel -> src pixel
    filled = np.where(np.isfinite(disp), disp, -1).astype(np.float32)
    out = cv2.warpPerspective(
        filled, h, (dst.out_w, dst.out_h), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT, borderValue=-1,
    )
    out[out <= 0] = np.nan
    return out


def fuse_disparities(primary, secondary, agree_px=1.5, conflict_px=3.0, min_agree=0.75):
    """Combine two disparity maps of the same view from different camera pairs.

    Both are affine in inverse depth, so they are related by  secondary = a*primary + b.
    That line is fitted where both are valid, the secondary is converted into the
    primary's units, and then per pixel:
      both agree      -> average, weighted by how sensitive each pair is to depth
      clear conflict  -> dropped (at least one is a false match)
      only one valid  -> that one (the pairs see around different occlusions)
    Returns (fused, info). If the line cannot be fitted, or less than min_agree of the
    overlap agrees (the secondary is then the unreliable one), the primary is returned
    unchanged.
    """
    va, vb = np.isfinite(primary), np.isfinite(secondary)
    both = va & vb
    info = {"overlap": int(both.sum()), "fused": False}
    a, b, inliers = geometry.fit_line(primary[both][::7], secondary[both][::7])
    if a is None or a <= 0:
        return primary, info
    converted = (secondary - b) / a
    converted[~(converted > 0)] = np.nan
    vb = np.isfinite(converted)
    both = va & vb
    diff = np.abs(primary - converted)
    agree = both & (diff < agree_px)
    if both.sum() < 500 or agree.sum() < min_agree * both.sum():
        # the secondary is the one that is off this frame; using it would punch holes
        info.update(agree=int(agree.sum()), overlap=int(both.sum()))
        return primary, info
    w = a * a / (1 + a * a)  # weight of the secondary: its noise shrinks by a when converted
    out = primary.copy()
    agree = both & (diff < agree_px)
    out[agree] = (1 - w) * primary[agree] + w * converted[agree]
    out[both & (diff > conflict_px)] = np.nan
    only_b = vb & ~va
    out[only_b] = converted[only_b]
    info.update(fused=True, scale=a, offset=b, agree=int(agree.sum()),
                dropped=int((both & (diff > conflict_px)).sum()), filled=int(only_b.sum()))
    return out, info


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


def display_range(disp: np.ndarray):
    """Disparity range that covers the scene (2nd to 98th percentile), for colouring."""
    valid = disp[np.isfinite(disp)]
    if valid.size < 100:
        return 0.0, 1.0
    lo, hi = np.percentile(valid, [2, 98])
    return float(lo), float(max(hi, lo + 1.0))


def colorize(disp: np.ndarray, lo: float | None = None, hi: float | None = None) -> np.ndarray:
    """Near = warm, far = cool, no data = black. The colour range follows the scene
    unless lo and hi (disparities) are given."""
    if lo is None or hi is None:
        lo, hi = display_range(disp)
    valid = np.isfinite(disp)
    norm = np.zeros(disp.shape, np.uint8)
    norm[valid] = np.clip((disp[valid] - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
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
