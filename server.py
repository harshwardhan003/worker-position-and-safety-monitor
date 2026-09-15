"""
PhantomMap 5.0 — Simple PDR Server
Just moves the dot. No zones, no calibration needed.
Pretrained HAR model for activity detection.
"""
import json, math, pickle, threading, time
import numpy as np
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

app  = Flask(__name__)
CORS(app)
lock = threading.Lock()

# ── Constants ──────────────────────────────────────────────────────────────────
ROOM_W = 9.0
ROOM_H = 8.0
STATE_FILE = "pm_state.json"

# ── Load pretrained HAR model ──────────────────────────────────────────────────
HAR_MODEL = None
try:
    HAR_MODEL = pickle.load(open("har_model.pkl", "rb"))
    print(f"[HAR] Pretrained model loaded — classes: {HAR_MODEL['classes']}")
except Exception as e:
    print(f"[HAR] No pretrained model: {e} — using physics fallback")

# ── PDR State ──────────────────────────────────────────────────────────────────
pos = {"x": 4.5, "y": 4.0}          # start in centre of room
heading   = 0.0                       # degrees
steps     = 0
activity  = "standing"
path_pts  = []                        # list of (x,y) for trail
kf        = {"x": 4.5, "y": 4.0,
             "vx": 0.0, "vy": 0.0,
             "px": 5.0, "py": 5.0}
sample_buf = []                       # rolling sensor window
step_count_last = 0
alert_log = []

# ── Vision state (posted by vision_engine.py) ─────────────────────────────────
vision_state = {
    "people_on_floor": 0,
    "time_on_floor": 0,
    "running_detected": False,
    "entry_count": 0,
    "exit_count": 0,
    "fatigue_score": 0,
    "frame_b64": "",
    "alerts": [],
    "updated": "--",
}

state = {
    "x": 4.5, "y": 4.0,
    "heading": 0.0,
    "activity": "standing",
    "steps": 0,
    "path": [],
    "updated": "--",
}

def kalman_update(mx, my):
    kf["px"] += 0.05; kf["py"] += 0.05
    kx = kf["px"] / (kf["px"] + 1.2)
    ky = kf["py"] / (kf["py"] + 1.2)
    kf["vx"] = 0.8*kf["vx"] + 0.3*(mx - kf["x"])
    kf["vy"] = 0.8*kf["vy"] + 0.3*(my - kf["y"])
    kf["x"] += kx*(mx - kf["x"])
    kf["y"] += ky*(my - kf["y"])
    kf["px"] *= (1-kx); kf["py"] *= (1-ky)
    return round(kf["x"],3), round(kf["y"],3)

def quaternion_to_yaw(qx, qy, qz, qw):
    siny = 2.0*(qw*qz + qx*qy)
    cosy = 1.0 - 2.0*(qy*qy + qz*qz)
    return math.degrees(math.atan2(siny, cosy))

def extract_har_features(samples):
    """Extract 10 features from sensor window for HAR model."""
    if not samples or len(samples) < 3:
        return None
    ax = [s.get("acc",[0,0,0])[0] for s in samples]
    ay = [s.get("acc",[0,0,0])[1] for s in samples]
    az = [s.get("acc",[0,0,0])[2] for s in samples]
    gz = [s.get("gyro",[0,0,0])[2] for s in samples]
    mag = [math.sqrt(x**2+y**2+z**2) for x,y,z in zip(ax,ay,az)]
    jerk= [abs(mag[i]-mag[i-1]) for i in range(1,len(mag))]
    return [
        float(np.mean(ax)),
        float(np.mean(ay)),
        float(np.mean(az)),
        float(np.std(ax)),
        float(np.std(ay)),
        float(np.std(az)),
        float(np.mean(jerk)) if jerk else 0.0,
        float(np.mean(gz)),
        float(np.mean(mag)**2) if mag else 0.0,
        float(np.corrcoef(ax,ay)[0,1]) if len(ax)>1 else 0.0,
    ]

def classify_activity(samples):
    """Use pretrained HAR model or physics fallback."""
    if HAR_MODEL and len(samples) >= 5:
        try:
            feats = extract_har_features(samples)
            if feats:
                f = np.array(feats).reshape(1,-1)
                fs= HAR_MODEL["scaler"].transform(f)
                return HAR_MODEL["rf"].predict(fs)[0]
        except Exception:
            pass
    # Physics fallback
    if len(samples) < 2:
        return "standing"
    mag = [math.sqrt(sum(v**2 for v in s.get("acc",[0,0,0]))) for s in samples]
    std = float(np.std(mag))
    gz  = abs(float(np.mean([s.get("gyro",[0,0,0])[2] for s in samples])))
    if gz > 0.4:   return "turning"
    if std > 0.3:  return "walking"
    return "standing"

def write_state():
    with lock:
        s = dict(state)
        s["path"] = path_pts[-300:]
    Path(STATE_FILE).write_text(json.dumps(s))

# ── Sensor stream endpoint (motion mode scanner) ──────────────────────────────
@app.route("/api/sensors", methods=["POST"])
def receive_sensors():
    global heading, steps, activity, step_count_last
    data    = request.json or {}
    samples = data.get("samples", [])
    wifi    = data.get("readings", {})

    if not samples:
        return jsonify({"ok": False})

    # Add to rolling buffer
    sample_buf.extend(samples)
    if len(sample_buf) > 100:
        del sample_buf[:-100]

    # 1. Classify activity
    activity = classify_activity(sample_buf[-25:])

    # 2. Heading from rotation vector if present, else integrate gyro
    last = samples[-1]
    rot  = last.get("rot")
    if rot and len(rot) >= 4:
        new_yaw  = quaternion_to_yaw(rot[0], rot[1], rot[2], rot[3])
        heading  = heading * 0.85 + new_yaw * 0.15
    else:
        gz_vals  = [s.get("gyro",[0,0,0])[2] for s in samples]
        gz_mean  = float(np.mean(gz_vals))
        dt       = len(samples) * 0.04   # ~25Hz
        heading += math.degrees(gz_mean * dt)

    # 3. Step detection from accelerometer peaks
    step_happened = False
    if data.get("step"):
        step_happened = True
        steps += 1
    else:
        # Fallback: detect steps from accel magnitude peaks
        mag_vals = [math.sqrt(sum(v**2 for v in s.get("acc",[0,0,0])))
                    for s in samples]
        threshold = 1.8
        for i in range(1, len(mag_vals)-1):
            if (mag_vals[i] > threshold and
                mag_vals[i] > mag_vals[i-1] and
                mag_vals[i] > mag_vals[i+1]):
                step_happened = True
                steps += 1
                break

    # 4. Move position if walking + step detected
    if step_happened and activity in ("walking", "turning"):
        # Weinberg stride estimate
        mag_all = [math.sqrt(sum(v**2 for v in s.get("acc",[0,0,0])))
                   for s in sample_buf[-15:]]
        if mag_all:
            diff   = max(mag_all) - min(mag_all)
            stride = 0.45 * (diff ** 0.25) if diff > 0 else 0.65
            stride = max(0.3, min(1.2, stride))
        else:
            stride = 0.65

        hr = math.radians(heading)
        nx = pos["x"] + stride * math.cos(hr)
        ny = pos["y"] + stride * math.sin(hr)
        # Clamp to room
        pos["x"] = max(0.2, min(ROOM_W-0.2, nx))
        pos["y"] = max(0.2, min(ROOM_H-0.2, ny))

    # 5. Kalman smooth
    sx, sy = kalman_update(pos["x"], pos["y"])
    path_pts.append({"x": sx, "y": sy})
    if len(path_pts) > 500:
        path_pts.pop(0)

    with lock:
        state.update({
            "x": sx, "y": sy,
            "heading": round(heading, 1),
            "activity": activity,
            "steps": steps,
            "updated": datetime.now().strftime("%H:%M:%S"),
        })
    write_state()

    return jsonify({
        "pdr": {"steps": steps, "activity": activity,
                "heading_deg": round(heading,1)},
        "position": {"x": sx, "y": sy, "radius": 0.5},
    })

# ── WiFi scan fallback ─────────────────────────────────────────────────────────
@app.route("/api/scan", methods=["POST"])
def receive_scan():
    # Accept WiFi-only scans too — just return current position
    return jsonify({"x": pos["x"], "y": pos["y"], "zone": "tracking",
                    "conf": 0.8, "breach": False})

# ── Reset start position ───────────────────────────────────────────────────────
@app.route("/api/reset_pos", methods=["POST"])
def reset_pos():
    data = request.json or {}
    pos["x"] = float(data.get("x", 4.5))
    pos["y"] = float(data.get("y", 4.0))
    kf["x"]  = pos["x"]; kf["y"] = pos["y"]
    path_pts.clear()
    with lock:
        state.update({"x":pos["x"],"y":pos["y"],"path":[],"steps":0})
    steps_ref = 0
    write_state()
    return jsonify({"x":pos["x"],"y":pos["y"]})


# ── Vision endpoint ────────────────────────────────────────────────────────────
@app.route("/api/vision", methods=["POST"])
def receive_vision():
    global vision_state
    data = request.json or {}
    with lock:
        vision_state.update(data)
    return jsonify({"ok": True})

# ── State ──────────────────────────────────────────────────────────────────────
@app.route("/api/state", methods=["GET"])
def get_state():
    with lock:
        s = dict(state)
    s["path"] = path_pts[-300:]
    with lock:
        s["vision"] = dict(vision_state)
    return jsonify(s)

if __name__ == "__main__":
    write_state()
    print(f"[Server] PhantomMap 5.0 on :5050")
    print(f"[Server] HAR model: {'loaded' if HAR_MODEL else 'using physics'}")
    app.run(host="0.0.0.0", port=5050,
            debug=False, use_reloader=False, threaded=True)
