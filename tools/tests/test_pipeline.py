import numpy as np
import pytest

from mcdepth import geometry, synth
from mcdepth.pipeline import StereoDepth, StereoRig


def region_error(depth, truth, lo, hi):
    sel = np.isfinite(depth) & (truth > lo) & (truth < hi)
    assert sel.mean() > 0.02, "almost no valid depth in region"
    return abs(np.median(depth[sel]) / np.median(truth[sel]) - 1)


def run(rig, refine=False, **scene):
    ref, aux, truth = synth.render_pair(rig, **scene)
    sd = StereoDepth(rig, out_size=(640, 480))
    info = sd.refine(ref, aux) if refine else None
    rect_ref, _, disp, depth = sd.compute(ref, aux)
    # ground truth in the rectified reference view (rotation is tiny; reuse the map)
    import cv2

    truth_rect = cv2.remap(truth.astype(np.float32), *sd.map_ref, cv2.INTER_NEAREST)
    return depth, truth_rect, info, sd


@pytest.mark.parametrize(
    "baseline",
    [(0.012, 0, 0), (-0.012, 0, 0), (0.001, 0.012, 0), (0, -0.012, 0.001)],
    ids=["aux-right", "aux-left", "aux-below", "aux-above"],
)
def test_metric_depth_all_layouts(baseline):
    depth, truth, _, sd = run(synth.phone_rig(baseline=baseline, tilt_deg=0.4))
    assert sd.rect.vertical == (abs(baseline[1]) > abs(baseline[0]))
    assert region_error(depth, truth, 0.4, 0.6) < 0.04
    assert region_error(depth, truth, 1.3, 1.7) < 0.08
    assert np.isfinite(depth).mean() > 0.5


def test_refine_recovers_bad_extrinsics():
    true_rig = synth.phone_rig(tilt_deg=0.0)
    ref, aux, truth = synth.render_pair(true_rig)
    # the calibration the phone *reports* is off by about half a degree
    import cv2

    err, _ = cv2.Rodrigues(np.deg2rad([0.5, 0.1, 0.3]))
    bad = StereoRig(true_rig.ref, true_rig.aux, err @ true_rig.r, err @ true_rig.t)

    sd = StereoDepth(bad, out_size=(640, 480))
    _, _, _, before = sd.compute(ref, aux)
    info = sd.refine(ref, aux)
    assert info["applied"] and info["inliers"] > 50
    _, _, _, after = sd.compute(ref, aux)
    truth_rect = cv2.remap(truth.astype(np.float32), *sd.map_ref, cv2.INTER_NEAREST)
    assert np.isfinite(after).mean() > np.isfinite(before).mean() + 0.2
    assert region_error(after, truth_rect, 0.4, 0.6) < 0.08


def test_rectify_geometry():
    rng = np.random.default_rng(0)
    rig = synth.phone_rig(baseline=(-0.002, 0.011, 0.001), tilt_deg=1.5)
    rect = geometry.rectify(rig.r, rig.t)
    assert rect.vertical and rect.baseline > 0
    pts = rng.uniform([-0.3, -0.3, 0.3], [0.3, 0.3, 3.0], (50, 3))
    a = pts @ rect.r1.T
    b = (pts @ rig.r.T + rig.t) @ rect.r2.T
    np.testing.assert_allclose(a[:, 2], b[:, 2], atol=1e-12)  # same depth
    np.testing.assert_allclose(a[:, 0], b[:, 0], atol=1e-12)  # same column
    np.testing.assert_allclose(a[:, 1] - b[:, 1], rect.baseline, atol=1e-12)


def test_relative_pose_from_android_quaternions():
    # both cameras axis-aligned with each other, sensor frame = camera frame flipped about x
    q = (1.0, 0.0, 0.0, 0.0)
    r, t = geometry.relative_pose(q, (0, 0, 0), q, (0.01, 0, 0))
    np.testing.assert_allclose(r, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(t, [-0.01, 0, 0], atol=1e-12)


def test_scale_intrinsics_crops_to_aspect():
    fx, fy, cx, cy = geometry.scale_intrinsics(3000, 3000, 2000, 1500, 4000, 3000, 1280, 720)
    assert fx == pytest.approx(960) and cx == pytest.approx(640) and cy == pytest.approx(360)


def _depth_ok(sd, ref, aux, truth):
    import cv2

    _, _, _, depth = sd.compute(ref, aux)
    truth_rect = cv2.remap(truth.astype(np.float32), *sd.map_ref, cv2.INTER_NEAREST)
    return depth, truth_rect


def test_refine_detects_reversed_baseline():
    true_rig = synth.phone_rig()
    ref, aux, truth = synth.render_pair(true_rig)
    wrong = StereoRig(true_rig.ref, true_rig.aux, true_rig.r, -true_rig.t)  # phone reports the mirror layout
    sd = StereoDepth(wrong, out_size=(640, 480))
    info = sd.refine(ref, aux)
    assert info["applied"] and info["baseline_flipped"] and not sd.self_calibrated
    depth, truth_rect = _depth_ok(sd, ref, aux, truth)
    assert region_error(depth, truth_rect, 0.4, 0.6) < 0.08


def test_refine_survives_grossly_wrong_calibration():
    import cv2
    from mcdepth.pipeline import Camera

    true_rig = synth.phone_rig()
    ref, aux, truth = synth.render_pair(true_rig)
    k = true_rig.aux.k.copy()
    k[0, 0] *= 1.12  # focal length 12 % off
    k[1, 1] *= 1.12
    k[0, 2] += 25  # principal point 25 px off
    k[1, 2] -= 18
    err, _ = cv2.Rodrigues(np.deg2rad([0.0, 0.0, 3.0]))  # and rotated 3 degrees
    bad = StereoRig(true_rig.ref, Camera(1280, 960, k), err @ true_rig.r, err @ true_rig.t)
    sd = StereoDepth(bad, out_size=(640, 480))
    _, _, _, before = sd.compute(ref, aux)
    info = sd.refine(ref, aux)
    assert info["applied"] and info.get("coarse") and sd.self_calibrated
    depth, truth_rect = _depth_ok(sd, ref, aux, truth)
    assert np.isfinite(depth).mean() > max(0.5, np.isfinite(before).mean() + 0.2)
    near = np.nanmedian(depth[(truth_rect > 0.4) & (truth_rect < 0.6)])
    far = np.nanmedian(depth[(truth_rect > 1.3) & (truth_rect < 1.7)])
    assert near < far  # relative depth: ordering is right even though scale is not
