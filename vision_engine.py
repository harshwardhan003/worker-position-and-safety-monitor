"""
SentinelFloor — Vision Engine
Runs on laptop webcam, watches the doorway.
Detects people entering/exiting, counts time on floor,
detects running vs walking, posts to server every second.

Run separately: py -3.11 vision_engine.py
"""
import cv2, time, math, base64, json, requests, threading
import numpy as np
from datetime import datetime
from collections import deque
from pathlib import Path

try:
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    print("[Vision] YOLOv8n loaded")
except Exception as e:
    model = None
    print(f"[Vision] YOLOv8 not available: {e}")

SERVER = "http://127.0.0.1:5050"
VISION_FILE = "pm_vision.json"

# ── State ──────────────────────────────────────────────────────────────────────
people_on_floor = 0
session_start   = None
entry_times     = {}        # track_id → entry_time
exit_log        = []        # list of session dicts
prev_centroids  = {}        # track_id → deque of (x, frame_time)
alerts          = []

vision_state = {
    "people_on_floor": 0,
    "time_on_floor":   0,
    "running_detected": False,
    "entry_count":     0,
    "exit_count":      0,
    "fatigue_score":   0,
    "frame_b64":       "",
    "alerts":          [],
    "updated":         "--",
}

def encode_frame(frame, quality=55):
    _, buf = cv2.imencode(".jpg", frame,
                          [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode()

def estimate_speed(track_id, cx, cy, now, frame_w):
    """Estimate pixels/sec movement speed for a track."""
    if track_id not in prev_centroids:
        prev_centroids[track_id] = deque(maxlen=8)
    prev_centroids[track_id].append((cx, cy, now))
    if len(prev_centroids[track_id]) < 3:
        return 0.0
    oldest = prev_centroids[track_id][0]
    dt = now - oldest[2]
    if dt < 0.01:
        return 0.0
    dist = math.hypot(cx - oldest[0], cy - oldest[1])
    # Normalise by frame width → metres/sec estimate
    speed_px = dist / dt
    speed_norm = speed_px / frame_w   # 0-1 normalised
    return speed_norm

def post_vision(state):
    try:
        requests.post(f"{SERVER}/api/vision",
                      json=state, timeout=1)
    except Exception:
        pass

def run():
    global people_on_floor, session_start

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_count   = 0
    last_post     = 0
    entry_line_y  = None   # horizontal line across doorway
    running_frames= 0
    fatigue_scores= deque(maxlen=30)
    entry_count   = 0
    exit_count    = 0
    tracked_ids   = set()
    prev_detected = 0

    print("[Vision] Webcam started. Point at doorway/entrance.")
    print("[Vision] Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        h, w = frame.shape[:2]
        frame_count += 1
        now = time.time()

        if entry_line_y is None:
            entry_line_y = h // 2   # midpoint = doorway line

        annotated = frame.copy()

        # Draw entry line
        cv2.line(annotated, (0, entry_line_y),
                 (w, entry_line_y), (88, 166, 255), 2)
        cv2.putText(annotated, "ENTRY LINE",
                    (8, entry_line_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (88, 166, 255), 1)

        persons       = []
        running_now   = False
        current_count = 0

        if model and frame_count % 2 == 0:   # run every other frame
            results = model.track(frame, persist=True,
                                  classes=[0],    # person only
                                  conf=0.4,
                                  verbose=False)

            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    x1,y1,x2,y2 = map(int, box.xyxy[0])
                    cx = (x1+x2)//2
                    cy = (y1+y2)//2
                    tid = int(box.id[0]) if box.id is not None else -1
                    current_count += 1

                    # Speed estimate
                    speed = estimate_speed(tid, cx, cy, now, w)
                    is_running = speed > 0.18
                    if is_running:
                        running_now = True
                        running_frames += 1

                    # Entry/exit detection via line crossing
                    if tid not in tracked_ids:
                        tracked_ids.add(tid)
                        if cy < entry_line_y:
                            entry_count += 1
                            entry_times[tid] = now
                            if session_start is None:
                                session_start = now

                    # Bounding box colour
                    col = (0,80,220) if is_running else (0,200,100)
                    cv2.rectangle(annotated, (x1,y1), (x2,y2), col, 2)

                    # Blur face (top 28% of box)
                    fy2 = y1 + int((y2-y1)*0.28)
                    if fy2 > y1:
                        face = annotated[y1:fy2, x1:x2]
                        if face.size > 0:
                            annotated[y1:fy2, x1:x2] = \
                                cv2.GaussianBlur(face, (31,31), 0)

                    # Speed label
                    label = f"RUNNING! {speed:.2f}" if is_running \
                            else f"walking {speed:.2f}"
                    cv2.putText(annotated, label, (x1, y1-6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)
                    persons.append({"tid":tid,"cx":cx,"cy":cy,"speed":speed})

        # Exit detection — person disappeared
        if current_count < prev_detected and prev_detected > 0:
            exit_count += 1

        prev_detected     = current_count
        people_on_floor   = current_count

        # Time on floor
        time_on_floor = int(now - session_start) if session_start else 0

        # Fatigue: walking speed declining over time
        if persons:
            avg_speed = sum(p["speed"] for p in persons) / len(persons)
            fatigue_scores.append(avg_speed)
        fatigue_score = 0
        if len(fatigue_scores) >= 10:
            early = list(fatigue_scores)[:5]
            late  = list(fatigue_scores)[-5:]
            if sum(early)/5 > 0:
                ratio = sum(late)/5 / (sum(early)/5)
                fatigue_score = max(0, int((1-ratio)*100))

        # Build alerts
        current_alerts = []
        if running_now:
            current_alerts.append({
                "type":"running",
                "msg":"⚠️ Running detected — safety violation",
                "ts": datetime.now().strftime("%H:%M:%S")
            })
        if time_on_floor > 300 and people_on_floor > 0:
            current_alerts.append({
                "type":"time",
                "msg":f"⏱️ Worker on floor {time_on_floor//60}m {time_on_floor%60}s",
                "ts": datetime.now().strftime("%H:%M:%S")
            })
        if fatigue_score > 60:
            current_alerts.append({
                "type":"fatigue",
                "msg":f"😴 Fatigue score {fatigue_score}/100",
                "ts": datetime.now().strftime("%H:%M:%S")
            })

        # HUD overlay
        hud_col = (40, 15, 15) if current_alerts else (15, 30, 15)
        cv2.rectangle(annotated, (0,0), (w,38), hud_col, -1)
        status = "⚠ ALERT" if current_alerts else "✓ CLEAR"
        cv2.putText(annotated,
            f"SentinelFloor  |  {status}  |  "
            f"{people_on_floor} person(s)  |  "
            f"Floor time: {time_on_floor}s  |  "
            f"Fatigue: {fatigue_score}/100",
            (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
            (220,220,220), 1)

        # Post to server every 1s
        if now - last_post >= 1.0:
            last_post = now
            vs = {
                "people_on_floor":  people_on_floor,
                "time_on_floor":    time_on_floor,
                "running_detected": running_now,
                "entry_count":      entry_count,
                "exit_count":       exit_count,
                "fatigue_score":    fatigue_score,
                "frame_b64":        encode_frame(annotated),
                "alerts":           current_alerts,
                "updated":          datetime.now().strftime("%H:%M:%S"),
            }
            Path(VISION_FILE).write_text(json.dumps(vs))
            threading.Thread(target=post_vision,
                             args=(vs,), daemon=True).start()

        cv2.imshow("SentinelFloor Vision (Q=quit)", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()
