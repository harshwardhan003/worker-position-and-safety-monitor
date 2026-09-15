# Worker Position and Safety Monitor

Indoor positioning and factory floor safety monitoring, combining phone sensor based dead reckoning with computer vision, built with a Digital Manufacturing Ireland use case in mind.

## What this is

The project has two cooperating parts that share a single Flask server.

**Position tracking** follows a person's location inside a room using their phone's accelerometer, gyroscope, and rotation vector, no beacons, no calibration, no fixed infrastructure. It fuses step detection, heading estimation, and a lightweight Kalman filter to keep a live dot moving smoothly on a floor plan as the person walks around.

**Floor safety monitoring** watches a doorway or work area through a laptop webcam using YOLOv8 person detection. It counts people entering and exiting, tracks how long someone has been on the floor, flags running as a safety violation, and estimates a fatigue score from declining walking speed over time. Faces are blurred automatically in the annotated feed.

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
| `server.py` | Flask server on port 5050. Runs the PDR pipeline (step detection, heading fusion, Kalman smoothing, optional HAR model for activity classification) and exposes the shared state to the dashboard. |
| `scanner.py` | Runs on the tracked device (phone via Termux, or a Windows laptop). Streams motion sensor data or WiFi RSSI readings to the server depending on `--mode`. |
| `vision_engine.py` | Runs on a laptop with a webcam. Detects and tracks people with YOLOv8, estimates movement speed, flags running and fatigue, blurs faces, and posts a live annotated frame to the server. |
| `pm_state.json` | Live position, heading, activity, and path trail, written by the server. |
| `pm_vision.json` | Live floor safety state: people on floor, time on floor, fatigue score, alerts, and the current annotated frame. |

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

## Key techniques

- **Pedestrian dead reckoning**: step detection from accelerometer peak picking, heading from quaternion to yaw conversion (falling back to gyroscope integration), and a Weinberg style stride length estimate from acceleration variance.
- **Sensor fusion**: a lightweight Kalman filter smooths raw position estimates into a stable trail.
- **Activity classification**: a pretrained HAR (Human Activity Recognition) model classifies walking, standing, and turning from a rolling window of sensor features, with a physics based rule fallback when no model is available.
- **Person tracking and safety inference**: YOLOv8 with persistent tracking IDs, entry and exit line crossing logic, per track speed estimation, and a fatigue score derived from the ratio of early session to late session walking speed.
- **Privacy by default**: faces are Gaussian blurred in every annotated frame before it is stored or transmitted.

## Notes

This is a working prototype built for experimentation with sensor fusion and vision based safety monitoring, not a production deployment. Position estimates drift over time without periodic recalibration, and the fatigue score is a heuristic, not a validated physiological measure.
