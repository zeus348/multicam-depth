# Recording format (version 1)

One directory per recording, written by the app to
`Android/data/dev.multicamdepth/files/session_YYYYMMDD_HHMMSS/`.

| File | Content |
|---|---|
| `ref/NNNNNN.jpg`, `aux/NNNNNN.jpg` | Frame pair `NNNNNN`, as delivered by the sensor (not rotated, not rectified). `ref` is the narrower lens. |
| `frames.csv` | `index,t_ref_ns,t_aux_ns,ref_file,aux_file`. Timestamps are each sensor's own start-of-exposure time (`Image.getTimestamp()`). |
| `imu.csv` | `t_ns,sensor,x,y,z` with `sensor` = `accel` (m/s²) or `gyro` (rad/s), at the fastest rate the phone allows. |
| `calibration.json` | See below. |

## calibration.json

```jsonc
{
  "format": 1,
  "device": "Google Pixel 8",
  "sensor_orientation": 90,        // rotate images clockwise by this to view them upright
  "sync_type": 1,                  // 1: sensors are hardware synced, 0: approximately
  "timestamp_source": 1,           // 1: camera and IMU timestamps share one clock, 0: unknown offset
  "ref": { "id": "2", "width": 1280, "height": 960,
           "intrinsics": [fx, fy, cx, cy],          // pixels of the recorded images
           "distortion": [k1, k2, p1, p2, k3] },    // OpenCV order; zeros if the phone already corrected it
  "aux": { ... },
  "stereo": { "R": [9 numbers, row-major], "T": [x, y, z],   // X_aux = R * X_ref + T, metres
              "metric": true },    // false: the phone published no camera positions, baseline is a guess
  "calibration_source": "...",     // which parts were published and which were derived
  "reported": { "ref": {...}, "aux": {...} }   // raw Camera2 values, for re-deriving the calibration
}
```

Camera frame: x right, y down, z forward, in the image as stored.
