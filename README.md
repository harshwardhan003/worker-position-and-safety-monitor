# Worker Position and Safety Monitor

Indoor positioning and factory floor safety monitoring, combining phone sensor based dead reckoning with computer vision, built with a Digital Manufacturing Ireland use case in mind.

## What this is

The project has two cooperating parts that share a single Flask server.

**Position tracking** follows a person's location inside a room using their phone's accelerometer, gyroscope, and rotation vector, no beacons, no calibration, no fixed infrastructure. It fuses step detection, heading estimation, and a lightweight Kalman filter to keep a live dot moving smoothly on a floor plan as the person walks around.

**Floor safety monitoring** watches a doorway or work area through a laptop webcam using a YOLOv8 object detection model. It counts people entering and exiting, tracks how long someone has been on the floor, flags running as a safety violation, and estimates a fatigue score from declining walking speed over time. Faces are blurred automatically in the annotated feed.

Both systems post their state to a shared server, so a single dashboard can show live position, activity, and floor safety alerts together.

## Why it is built this way

Most indoor positioning demos assume you can install beacons, run a calibration walk, or rely on WiFi fingerprinting that Android throttles into uselessness after a few scans. This system avoids all three: it uses pedestrian dead reckoning (PDR) from onboard motion sensors alone, with WiFi scanning kept as an optional, non essential fallback rather than the primary signal.

Floor safety monitoring treats safety inference as a webcam problem, not a specialised hardware problem. YOLOv8 already solves person detection and tracking reliably enough that the harder work is turning raw detections into operationally useful signals: dwell time, entry and exit counts, and a fatigue heuristic derived from movement speed decay.

## Architecture

```
Phone (Termux)                    Laptop webcam
     |                                  |
     | accelerometer, gyroscope,        | YOLOv8 person detection
     | rotation vector                  | + speed/fatigue estimation
     v                                  v
scanner.py  ----POST /api/sensors--->  server.py  <---POST /api/vision----  vision_engine.py
                                          |
                                          | writes
                                          v
                                  pm_state.json / pm_vision.json
                                          |
                                          v
                                     dashboard (Streamlit, :8501)
```

## Components

| File | Role |
|---|---|
| `run.py` | Single entry point. Installs missing dependencies, starts the server and dashboard, opens the browser. |
| `server.py` | Flask server on port 5050. Runs the PDR pipeline (step detection, heading fusion, Kalman smoothing, HAR classification) and exposes shared state to the dashboard. |
| `scanner.py` | Runs on the tracked device (phone via Termux, or a Windows laptop). Streams motion sensor data or WiFi RSSI readings to the server depending on `--mode`. |
| `vision_engine.py` | Runs on a laptop with a webcam. Detects and tracks people with YOLOv8, estimates movement speed, flags running and fatigue, blurs faces, and posts a live annotated frame to the server. |
| `pm_state.json` | Live position, heading, activity, and path trail, written by the server. |
| `pm_vision.json` | Live floor safety state: people on floor, time on floor, fatigue score, alerts, and the current annotated frame. |

## The models and algorithms, in depth

This section explains what each model actually does internally, not just what it is called.

### 1. Human Activity Recognition (HAR) classifier

**Problem it solves:** deciding whether the phone carrier is standing, walking, or turning, which gates whether the position estimate should move at all.

**Model type:** a scikit-learn `RandomForestClassifier` paired with a fitted `StandardScaler`, both serialised together into `har_model.pkl`. Random forests were chosen over a deep model here because the feature set is small (10 features), the classes are simple and well separated physically, and a forest gives fast, low-latency inference on a phone-class CPU with no GPU dependency.

**Feature extraction** (`extract_har_features`): for each rolling window of recent sensor samples, 10 features are computed:

1. Mean acceleration on X, Y, Z axes (3 features) - captures orientation-dependent gravity bias and steady drift.
2. Standard deviation of acceleration on X, Y, Z (3 features) - the core walking signal: standing has low variance, walking has a strong periodic oscillation, so std deviation spikes.
3. Mean jerk, i.e. the mean absolute frame-to-frame change in acceleration magnitude - distinguishes the sharp, high-frequency impacts of footsteps from smoother motion.
4. Mean gyroscope Z-axis rate - the primary turning signal; high angular velocity around the vertical axis means the person is rotating in place rather than walking straight.
5. Mean squared acceleration magnitude - a coarse energy proxy for overall motion intensity.
6. Pearson correlation between X and Y acceleration - captures characteristic phase relationships in the arm/hip swing pattern that differ between gaits.

**Inference path:** the last 25 samples in the rolling buffer are scaled with the saved `StandardScaler` and passed to the forest, which returns one of the trained class labels directly. If the model file is missing, fails to load, or there are fewer than 5 samples buffered, the code falls back to a transparent physics rule: gyroscope Z mean above 0.4 rad/s means "turning", acceleration magnitude standard deviation above 0.3 means "walking", otherwise "standing". This fallback is intentionally simple and auditable rather than a second learned model, so the system degrades gracefully instead of failing silently.

### 2. Pedestrian dead reckoning (PDR) and sensor fusion

**Problem it solves:** turning a stream of raw accelerometer and gyroscope samples into an (x, y) position on a floor plan, with no external reference signal.

**Step detection:** implemented as peak detection over the accelerometer magnitude signal. A step is registered when a sample's combined 3-axis magnitude exceeds a fixed threshold (1.8) and is a local maximum relative to its immediate neighbours. This is the same basic principle used in most consumer step counters, deliberately simple so it runs in real time on every sensor batch.

**Heading estimation:** two paths, chosen automatically based on what data is available in a sample.

- If a rotation vector (quaternion) is present, heading is derived by converting the quaternion to yaw with `atan2(2(qw*qz + qx*qy), 1 - 2(qy^2 + qz^2))`, then blended into the running heading estimate with an exponential moving average (85 percent previous value, 15 percent new reading) to suppress jitter from sensor noise.
- If no rotation vector is available, heading is estimated by integrating the gyroscope's Z-axis angular velocity over the elapsed time of the batch, a standard dead-reckoning fallback that will drift over long sessions but works well for short walks.

**Stride length estimation:** rather than assuming a fixed step length, stride length is estimated per step using a Weinberg-style formula: `stride = 0.45 * (max(accel_magnitude) - min(accel_magnitude)) ** 0.25`, clamped between 0.3 and 1.2 metres. The intuition is that a more vigorous vertical acceleration swing during a step correlates with a longer stride, so this adapts to walking pace instead of using one constant.

**Position update:** each detected step moves the estimated position by `stride` metres in the direction of the current heading (`x += stride * cos(heading)`, `y += stride * sin(heading)`), then the result is clamped to stay within the configured room bounds (9m x 8m by default).

**Kalman filtering:** the raw stepped position is smoothed by a simplified 2D Kalman filter before being shown on the dashboard. The filter maintains a running position estimate and an error covariance (`px`, `py`) that grows slightly each update (process noise) and shrinks each time a new measurement is blended in, weighted by the Kalman gain `k = p / (p + measurement_noise)`. This produces a visibly smoother, less jittery trail than plotting the raw stepped position directly, without needing a full state-space model.

### 3. YOLOv8 person detection and tracking

**Problem it solves:** finding every person in a webcam frame, following each one across frames with a stable identity, and turning that into safety-relevant signals.

**Model:** `yolov8n.pt`, the nano variant of Ultralytics' YOLOv8 object detector, chosen specifically for its small size and fast inference so it can run on ordinary laptop hardware without a dedicated GPU. Detection is restricted to class 0 (person) only, with a confidence threshold of 0.4, and runs on every second frame to keep the pipeline real time.

**Tracking:** YOLOv8's built-in `track(persist=True)` mode is used rather than treating each frame's detections independently. This assigns a persistent track ID to each detected person across frames, which is what makes entry/exit counting and per-person speed estimation possible, without persistent IDs, the system would have no way to know if the person in frame 50 is the same person seen in frame 10.

**Speed estimation:** for each tracked ID, the centroid position (`cx`, `cy`) is stored in a rolling deque of up to 8 recent (x, y, timestamp) points. Speed is computed as the Euclidean pixel distance between the oldest and newest stored point, divided by the elapsed time, then normalised by frame width to get a resolution-independent speed value between roughly 0 and 1. A person is flagged as running when this normalised speed exceeds 0.18, a threshold tuned empirically rather than derived from a calibrated real-world speed.

**Entry and exit detection:** a horizontal line is drawn at the vertical midpoint of the frame. The first time a track ID is seen, if its centroid is above that line, it counts as an entry. Exits are inferred more crudely: when the total number of currently detected people drops compared to the previous frame, that is counted as one exit. This is a simple heuristic rather than a matched entry/exit pairing per individual, which is a known limitation noted below.

**Fatigue scoring:** the system keeps a rolling window of the last 30 average per-frame speed readings. Once at least 10 readings are available, it compares the mean speed of the earliest 5 readings in the window against the mean speed of the latest 5. The fatigue score is `max(0, (1 - late_avg/early_avg) * 100)`, so a worker moving at the same pace throughout scores near 0, while someone slowing down significantly over the session scores higher. This is a relative, session-based heuristic, not an absolute or medically validated fatigue measure.

**Privacy layer:** on every detected bounding box, the top 28 percent (approximating head and face region) is Gaussian blurred (`cv2.GaussianBlur`, kernel size 31) before the frame is encoded and transmitted, so no recognisable face data leaves the vision engine even transiently.

## Getting started

**1. Start the core system**

```bash
py -3.11 run.py
```

This installs any missing dependencies, starts `server.py` on port 5050, launches the dashboard on port 8501, and opens it in your browser.

**2. Track a person (choose one)**

On an Android phone, inside Termux with `termux-api` installed:

```bash
pkg install termux-api
python scanner.py --host <server-ip> --mode motion
```

On a Windows laptop, using WiFi signal fingerprinting instead of motion sensors:

```bash
py -3.11 scanner.py --host localhost --mode wifi
```

**3. Run floor safety monitoring (optional, separate process)**

```bash
py -3.11 vision_engine.py
```

Point the webcam at a doorway or work area. Press `Q` in the video window to stop.

## Known limitations

- Position estimates drift over time without periodic recalibration, since PDR has no absolute position reference; long sessions will accumulate error.
- The running speed threshold (0.18) and fatigue thresholds were tuned by observation, not calibrated against real-world walking speeds in metres per second.
- Exit counting is based on a drop in total detected people, not a per-individual matched exit event, so simultaneous entries and exits in the same frame can undercount.
- The fatigue score is a relative movement-speed heuristic and should not be treated as a physiological or medical measurement.
- All computer vision runs on a single local webcam feed; there is no multi-camera handoff if a person leaves one camera's field of view and enters another.

## Notes

This is a working prototype built for experimentation with sensor fusion and vision based safety monitoring, not a production deployment.
