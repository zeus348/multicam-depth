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


def test_self_calibration_on_placeholder_phone_calibration():
    """A phone that reports nominal values: lenses "parallel", baseline in the wrong
    direction, while really the second camera points a few degrees off, is rolled,
    and sits mostly below the first one."""
    import cv2

    rvec = np.deg2rad([1.5, -3.0, 2.5])
    r_true, _ = cv2.Rodrigues(rvec)
    c2_true = np.array([0.004, 0.018, 0.0])
    base = synth.phone_rig()
    true_rig = StereoRig(base.ref, base.aux, r_true, -r_true @ c2_true)
    ref, aux, truth = synth.render_pair(true_rig, near=0.35, far=0.9)

    reported = StereoRig(base.ref, base.aux, np.eye(3), -np.array([0.018, 0.010, 0.0]))
    sd = StereoDepth(reported, out_size=(640, 480))
    _, _, _, before = sd.compute(ref, aux)
    info = sd.refine(ref, aux)
    assert info["applied"] and info["self_calibrated"] and info["direction_measured"], info
    u = np.array(info["baseline_direction"])
    true_dir = c2_true[:2] / np.linalg.norm(c2_true[:2])
    assert np.degrees(np.arccos(np.clip(u @ true_dir, -1, 1))) < 8
    depth, truth_rect = _depth_ok(sd, ref, aux, truth)
    assert np.isfinite(depth).mean() > max(0.4, np.isfinite(before).mean() + 0.2)
    near = np.nanmedian(depth[(truth_rect > 0.3) & (truth_rect < 0.4)])
    far = np.nanmedian(depth[(truth_rect > 0.8) & (truth_rect < 1.0)])
    assert near < far  # relative depth: ordering is right even though scale is not


def test_orientation_rotation_matches_a_portrait_back_camera():
    r = geometry.orientation_rotation(90)
    np.testing.assert_allclose(r, [[0, -1, 0], [-1, 0, 0], [0, 0, -1]], atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(r), 1.0)
    np.testing.assert_allclose(np.linalg.det(geometry.orientation_rotation(270, front_facing=True)), 1.0)


def test_parallax_direction_ignores_stray_matches():
    rng = np.random.default_rng(5)
    u = np.array([np.cos(1.1), np.sin(1.1)])
    good = np.outer(rng.uniform(-40, 40, 80), u) + rng.normal(0, 0.3, (80, 2))
    stray = rng.uniform(-60, 60, (25, 2))
    d, n = geometry.parallax_direction(np.vstack([good, stray]))
    assert d is not None and n > 60 and abs(d @ u) > 0.995
    assert geometry.parallax_direction(rng.uniform(-60, 60, (100, 2)))[0] is None


def test_fit_line_is_robust():
    rng = np.random.default_rng(1)
    x = rng.uniform(2, 30, 500)
    y = 1.8 * x - 4 + rng.normal(0, 0.1, 500)
    y[::6] = rng.uniform(0, 60, len(y[::6]))
    a, b, n = geometry.fit_line(x, y)
    assert a == pytest.approx(1.8, abs=0.02) and b == pytest.approx(-4, abs=0.3) and n > 380
    assert geometry.fit_line(np.full(500, 5.0), y)[0] is None  # flat scene: slope unknowable


def test_gyro_integration_recovers_a_constant_turn():
    t = np.arange(0, 101) * 5_000_000  # 200 Hz for half a second
    rates = np.tile([0.0, 0.4, 0.0], (len(t), 1))
    r = geometry.integrate_gyro(t, rates, t[10], t[90])
    np.testing.assert_allclose(r, geometry.rodrigues([0, 0.4 * 0.4, 0]), atol=1e-9)
    np.testing.assert_allclose(geometry.integrate_gyro(t, rates, t[5], t[5]), np.eye(3))


def test_third_camera_fills_holes_and_removes_false_matches():
    from mcdepth.pipeline import fuse_disparities, warp_disparity

    wide, tele = synth.phone_rig3()
    ref, aux, truth = synth.render_pair(wide, near=0.45, far=1.2)
    aux2, _ = synth.render(tele.aux, tele.r, tele.t, near=0.45, far=1.2)
    a = StereoDepth(wide, out_size=(640, 480))
    b = StereoDepth(tele, out_size=(640, 480))
    _, _, disp_a, _ = a.compute(ref, aux)
    _, _, disp_b, _ = b.compute(ref, aux2)
    assert np.isfinite(disp_b).mean() > 0.5
    moved = warp_disparity(disp_b, b, a)
    centre = np.isfinite(moved)
    assert 0.1 < centre.mean() < 0.4  # the telephoto only sees the middle of the main view

    # damage the first pair the way real scenes do: a hole and a patch of false matches
    damaged = disp_a.copy()
    damaged[200:240, 280:340] = np.nan
    damaged[250:270, 300:360] += 12
    fused, info = fuse_disparities(damaged, moved)
    assert info["fused"] and info["filled"] >= 1500 and info["dropped"] >= 800
    expect = wide.ref.k[0, 0] / 2 * 0.01503 / 0.45  # true disparity of the near square
    assert np.nanmedian(fused[200:240, 280:340]) == pytest.approx(expect, abs=0.6)
    assert np.isnan(fused[250:270, 300:360]).mean() > 0.8
    # and it must not make the untouched part worse
    untouched = np.isfinite(disp_a) & np.isfinite(fused)
    assert np.nanmedian(np.abs(fused - disp_a)[untouched]) < 0.25
