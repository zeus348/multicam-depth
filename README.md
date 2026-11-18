# multicam-depth

Depth from the two (or more) rear cameras of one Android phone, used as a stereo pair.

- **`android/`** – app that streams two physical cameras at once, shows a live depth
  map, and records frame pairs with per-frame sensor timestamps and IMU data.
- **`tools/`** – Python package that turns a recording into depth maps and point
  clouds, using the same algorithm as the app at full resolution.

MIT licensed. Only dependency beyond the platform is OpenCV (Apache-2.0).

![Synthetic example: rectified reference image and depth map](docs/demo.jpg)

## Status

| Part | State |
|---|---|
| Geometry (pose, rectification, alignment fit) | Tested in Python and Kotlin |
| Depth pipeline (Python) | Tested end to end on rendered scenes with known depth |
| Android app | Compiles and builds; **not yet run on a phone** |

The camera, recording and on-device OpenCV code has had no device testing. Expect
to fix things on first run; the info panel in the app prints what the phone reports.

## How it works

1. **Find a pair.** The app lists logical multi-cameras and their physical cameras
   (Camera2 `setPhysicalCameraId`). The narrower lens becomes the reference.
2. **Calibration from the phone.** Intrinsics, distortion and lens poses are read from
   `CameraCharacteristics`. Missing intrinsics are derived from focal length and sensor
   size. Many phones publish placeholders (identical rotations, lens spacings of a
   millimetre or two); these are detected and only their direction is kept.
3. **Rectify.** Both images are warped so that a scene point lands on the same row
   (or column, for cameras stacked along the image's y axis) at the narrower lens's
   focal length.
4. **Align from image content.** ORB feature matches are used to fit and remove the
   leftover row misalignment, and to detect a camera order reported the wrong way round.
   If that lines up only a minority of the matches, the calibration is treated as
   unusable and the geometry is measured instead: the second camera's rotation and
   focal-length error from how the dominant surface maps between the images, and the
   baseline direction from the direction of parallax. Depth is then labelled relative,
   because the true lens spacing and the zero-disparity point are unknown. The app
   retries until an alignment succeeds.
5. **Track rows per frame.** While the phone moves the cameras drift a few pixels out
   of line: the sensors are not exposed at the same instant, their rolling shutters
   differ, and stabilisation or autofocus shifts one lens but not the other. Row-mean
   profiles, which do not depend on depth, are compared over a 3x3 grid of blocks and
   the fitted shift is removed before matching.
   Autofocus is locked once the cameras are aligned, because a refocusing lens shifts
   and rescales its image; Re-align refocuses first.
6. **Match.** OpenCV semi-global block matching gives disparity `d`; depth is
   `focal_px * baseline / d`. A 5x5 median removes thin streaks.
7. **Third camera.** With three cameras the middle lens is the reference and forms a
   pair with each of the others. Each pair's disparity is affine in inverse depth, so
   the two are related by a line, fitted where both are valid. Then per pixel:
   agreeing values are averaged, clear conflicts are dropped as false matches, and
   holes in one pair are filled from the other. The telephoto only covers the centre
   of the view. If the phone cannot run three sensors at once the app falls back to two.
8. **Gyroscope.** The previous depth frame is blended into the new one to suppress
   flicker. The gyroscope's rotation since that frame says where each pixel has moved
   to, so the blend holds while the phone turns. Alignment waits until the phone is
   held still. This needs camera and IMU timestamps on the same clock.

### What to expect

Phone cameras sit about 1 cm apart, so depth resolution falls off quickly. At
640 px wide with a main-camera lens, one pixel of disparity is roughly 5 m, so
usable depth is about 0.1 m to 2 m live, and about twice that from full-resolution
recordings. Textureless surfaces give no reading (shown black).

## Build the app

```bash
cd android
```

```bash
./gradlew assembleDebug
```

```bash
adb install app/build/outputs/apk/debug/app-debug.apk
```

Needs Android 9+ and a phone that exposes its physical cameras to third-party apps
(most Pixels, many recent Samsung, Xiaomi and OnePlus models). If none is found the
app says so and lists what the phone does expose.

In the app: pick a pair, point at something textured 0.2 to 1.5 m away. **Re-align**
redoes step 4 for the current scene. Tap the left image to switch between the
reference camera, the second camera and a red/cyan overlay of both; in a good
alignment the overlay is grey with colour fringes only along the baseline. **Record** writes a session; **Share last** zips it.

## Process a recording

```bash
adb pull /sdcard/Android/data/dev.multicamdepth/files/session_20261117_120000
```

```bash
pip install -e tools
```

```bash
python -m mcdepth.cli process session_20261117_120000 --ply
```

Writes `depth/NNNNNN_depth_mm.png` (16-bit millimetres), a side-by-side preview and
optional point clouds. No phone yet? `python -m mcdepth.cli demo demo_session` writes
a synthetic recording to try it on.

Recording layout: [docs/session-format.md](docs/session-format.md).

## Tests

```bash
pytest tools
```

```bash
cd android && ./gradlew testDebugUnitTest
```

## Known limits

- Timestamps come from each sensor; frames are paired when within 16 ms. Whether the
  sensors are hardware-synced is recorded as `sync_type`.
- Autofocus and optical stabilisation shift the real intrinsics. OIS is switched off;
  focus is left automatic.
- Recordings are JPEG sequences, not video; pairs are skipped (and counted) if
  storage cannot keep up.
- Assumes intrinsics refer to the full active array, centre-cropped to the stream.
