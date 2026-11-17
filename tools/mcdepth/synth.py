"""Synthetic two-camera renderer used by the tests and the demo session.

Scene: a textured wall at `far` metres with a textured square at `near` metres
in front of it, both facing the reference camera. Each camera is rendered by
casting a ray per pixel, so different intrinsics and relative rotation are exact.
"""
from __future__ import annotations

import cv2
import numpy as np

from .pipeline import Camera, StereoRig


def _texture(seed, size=2048):
    rng = np.random.default_rng(seed)
    tex = np.zeros((size, size), np.float32)
    for cells in (32, 128, 512):  # multi-scale noise so every zoom level has detail
        tex += cv2.resize(rng.random((cells, cells), dtype=np.float32), (size, size), interpolation=cv2.INTER_CUBIC)
    return cv2.normalize(tex, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def render(cam: Camera, r: np.ndarray, t: np.ndarray, near=0.5, far=1.5, half=0.12):
    """Image of the scene from a camera with pose X_cam = r @ X_world + t.
    Also returns the true depth map (z in this camera's frame)."""
    h, w = cam.height, cam.width
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    rays_cam = np.stack([(u - cam.k[0, 2]) / cam.k[0, 0], (v - cam.k[1, 2]) / cam.k[1, 1], np.ones_like(u)], -1)
    rays = rays_cam @ r  # world direction = r.T @ ray
    origin = -r.T @ t
    img = np.zeros((h, w), np.uint8)
    depth = np.full((h, w), np.nan)
    metres_per_texel = 0.0012
    for plane_z, seed, limit in ((far, 1, None), (near, 2, half)):
        s = (plane_z - origin[2]) / rays[..., 2]
        x = origin[0] + s * rays[..., 0]
        y = origin[1] + s * rays[..., 1]
        hit = s > 0
        if limit is not None:
            hit &= (np.abs(x) < limit) & (np.abs(y) < limit)
        tex = _texture(seed)
        mx = (x / metres_per_texel + tex.shape[1] / 2).astype(np.float32)
        my = (y / metres_per_texel + tex.shape[0] / 2).astype(np.float32)
        sampled = cv2.remap(tex, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        img[hit] = sampled[hit]
        depth[hit] = (s * rays_cam[..., 2])[hit]
    return img, depth


def phone_rig(width=1280, height=960, baseline=(0.012, 0.0, 0.0), tilt_deg=0.0) -> StereoRig:
    """Main (reference) + ultrawide (aux) pair roughly like a phone's."""
    def k(f):
        return np.array([[f, 0, width / 2 + 3], [0, f, height / 2 - 2], [0, 0, 1.0]])

    rvec = np.deg2rad(tilt_deg) * np.array([1.0, 0.3, 0.5])
    r, _ = cv2.Rodrigues(rvec)
    c2 = np.array(baseline, float)  # aux position in reference coordinates
    return StereoRig(
        ref=Camera(width, height, k(0.75 * width)),
        aux=Camera(width, height, k(0.42 * width)),
        r=r,
        t=-r @ c2,
    )


def render_pair(rig: StereoRig, **scene):
    ref, depth = render(rig.ref, np.eye(3), np.zeros(3), **scene)
    aux, _ = render(rig.aux, rig.r, rig.t, **scene)
    return ref, aux, depth
