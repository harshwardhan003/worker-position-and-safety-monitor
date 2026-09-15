"""
PhantomMap 3.0 — Device scanner

Runs on whatever is doing the walking and posts WiFi scans to the server.
One endpoint handles everything: the server decides whether a scan is a
calibration capture or a live position fix, so there are no modes to get wrong.

    # Android phone, in Termux
    python scanner.py --host 192.168.1.20

    # This Windows laptop (uses netsh)
    py -3.11 scanner.py --host localhost

    # No hardware at all — use the browser's "Virtual device" switch instead.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime

import requests

RSSI_UNSEEN = -100


# ══════════════════════════════════════════════════════════════════════════════
# Back-ends
# ══════════════════════════════════════════════════════════════════════════════

def scan_termux():
    out = subprocess.run(["termux-wifi-scaninfo"], capture_output=True,
                         text=True, timeout=12)
    readings = {}
    for n in json.loads(out.stdout or "[]"):
        ssid = (n.get("ssid") or "").strip()
        bssid = (n.get("bssid") or "").replace(":", "")
        if not bssid:
            continue
        key = f"{bssid[-6:]}_{ssid[:10]}" if ssid else bssid[-12:]
        readings[key] = int(n.get("rssi", RSSI_UNSEEN))
    return readings


def scan_windows():
    """
    Parse `netsh wlan show networks mode=bssid`.

    netsh reports signal as a percentage; the widely used conversion back to
    dBm is dBm = pct/2 - 100, which is accurate enough for fingerprinting
    because the model only ever compares readings against each other.
    """
    out = subprocess.run(
        ["netsh", "wlan", "show", "networks", "mode=bssid"],
        capture_output=True, text=True, timeout=20,
    ).stdout

    readings = {}
    ssid = ""
    bssid = None
    for line in out.splitlines():
        line = line.strip()
        m = re.match(r"^SSID\s+\d+\s*:\s*(.*)$", line)
        if m:
            ssid = m.group(1).strip()
            continue
        m = re.match(r"^BSSID\s+\d+\s*:\s*([0-9a-fA-F:]{17})$", line)
        if m:
            bssid = m.group(1).replace(":", "")
            continue
        m = re.match(r"^Signal\s*:\s*(\d+)%$", line)
        if m and bssid:
            pct = int(m.group(1))
            key = f"{bssid[-6:]}_{ssid[:10]}" if ssid else bssid[-12:]
            readings[key] = int(pct / 2 - 100)
            bssid = None
    return readings


def stream_sensors(host, port, rate_hz, wifi_every, wifi_scan):
    """
    Stream accelerometer + gyroscope from termux-sensor to the server.

    termux-sensor emits a continuous run of JSON objects rather than one per
    line, so the stream is decoded incrementally with raw_decode. Sensor keys
    are the device's own driver names ("LSM6DSO Accelerometer"), which is why
    they're matched by substring rather than exact name.
    """
    delay_ms = max(int(1000 / max(rate_hz, 1)), 10)
    url = f"http://{host}:{port}/api/sensors"

    proc = subprocess.Popen(
        ["termux-sensor", "-s", "accelerometer,gyroscope", "-d", str(delay_ms)],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )

    decoder = json.JSONDecoder()
    buf = ""
    batch = []
    t0 = time.perf_counter()
    last_send = t0
    last_wifi = 0.0
    sent = 0

    print("  streaming motion sensors — walk around\n")
    try:
        while True:
            chunk = proc.stdout.read(256)
            if not chunk:
                break
            buf += chunk

            # Pull every complete JSON object currently in the buffer.
            while True:
                buf = buf.lstrip()
                if not buf:
                    break
                try:
                    obj, end = decoder.raw_decode(buf)
                except ValueError:
                    break
                buf = buf[end:]

                acc = gyro = None
                for key, val in obj.items():
                    k = key.lower()
                    values = val.get("values") if isinstance(val, dict) else None
                    if not values or len(values) < 3:
                        continue
                    if "accel" in k and acc is None:
                        acc = values[:3]
                    elif "gyro" in k and gyro is None:
                        gyro = values[:3]

                if acc and gyro:
                    batch.append({"t": round(time.perf_counter() - t0, 4),
                                  "acc": acc, "gyro": gyro})

            now = time.perf_counter()
            if batch and now - last_send >= 0.5:
                payload = {"samples": batch}

                # Fold in a WiFi scan occasionally; harmless if it's useless.
                if wifi_every and now - last_wifi >= wifi_every:
                    last_wifi = now
                    try:
                        payload["readings"] = wifi_scan()
                    except Exception:
                        pass

                try:
                    r = requests.post(url, json=payload, timeout=5).json()
                    sent += len(batch)
                    st = r.get("pdr") or {}
                    pos = r.get("position")
                    line = (f"  {sent:6d} samples · steps {st.get('steps', 0):3d}"
                            f" · {st.get('activity', '?'):8s}"
                            f" · heading {st.get('heading_deg', 0):6.1f}°")
                    if pos:
                        line += f" · ({pos['x']:.2f}, {pos['y']:.2f}) ±{pos['radius']:.1f}m"
                    print(line)
                except Exception as e:
                    print(f"  server unreachable: {e}")

                batch = []
                last_send = now
    finally:
        proc.terminate()


def pick_backend(choice):
    if choice == "termux":
        return scan_termux, "termux-wifi-scaninfo"
    if choice == "windows":
        return scan_windows, "netsh"
    if shutil.which("termux-wifi-scaninfo"):
        return scan_termux, "termux-wifi-scaninfo (auto)"
    if sys.platform.startswith("win"):
        return scan_windows, "netsh (auto)"
    raise SystemExit(
        "No WiFi scanner available on this platform.\n"
        "Use Termux on Android, run this on Windows, or switch on the\n"
        "Virtual device in the browser to demo without hardware.")


# ══════════════════════════════════════════════════════════════════════════════
# Main loop
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="PhantomMap device scanner")
    ap.add_argument("--host", default="localhost", help="server IP or hostname")
    ap.add_argument("--port", type=int, default=5050)
    ap.add_argument("--interval", type=float, default=1.5)
    ap.add_argument("--source", default="auto",
                    choices=["auto", "termux", "windows"])
    ap.add_argument("--mode", default="wifi", choices=["wifi", "motion"],
                    help="wifi = RSSI fingerprinting, motion = step tracking")
    ap.add_argument("--rate", type=float, default=25.0,
                    help="motion mode: sensor samples per second")
    ap.add_argument("--wifi-every", type=float, default=10.0,
                    help="motion mode: seconds between optional WiFi scans (0=off)")
    args = ap.parse_args()

    scan, name = pick_backend(args.source)

    # On Windows "localhost" resolves to ::1 first; the server is IPv4-only, so
    # every request would stall ~2s waiting for that refusal before retrying.
    host = "127.0.0.1" if args.host in ("localhost", "::1") else args.host
    url = f"http://{host}:{args.port}/api/scan"

    print(f"\n  PhantomMap scanner")
    print(f"  mode    : {args.mode}")
    print(f"  backend : {name}")
    print(f"  server  : {host}:{args.port}")
    print(f"  Ctrl+C to stop\n")

    if args.mode == "motion":
        if not shutil.which("termux-sensor"):
            raise SystemExit(
                "termux-sensor not found. Motion mode needs Termux:API:\n"
                "  pkg install termux-api   (and the Termux:API app from F-Droid)")
        stream_sensors(host, args.port, args.rate, args.wifi_every, scan)
        return

    n = 0
    last_sig = None
    repeats = 0
    warned = False

    while True:
        try:
            readings = scan()
        except Exception as e:
            print(f"  scan failed: {e}")
            time.sleep(args.interval)
            continue

        # Android throttles WiFi scanning and then keeps returning the SAME
        # cached result. That silently freezes positioning and poisons
        # calibration, so say so loudly rather than posting duplicates.
        sig = tuple(sorted(readings.items()))
        if sig == last_sig:
            repeats += 1
        else:
            repeats = 0
            warned = False
        last_sig = sig

        if repeats >= 3 and not warned:
            warned = True
            print("\n  !! WiFi scans are not changing — the phone is returning a")
            print("     cached scan. Positioning cannot work like this.")
            print("     Fix: Settings > About phone > tap Build number 7x >")
            print("          Developer options > turn OFF 'Wi-Fi scan throttling'")
            print(f"     Or restart with a slower interval:  --interval 30\n")

        if not readings:
            print("  no networks visible — is WiFi on?")
            time.sleep(args.interval)
            continue

        try:
            r = requests.post(url, timeout=5, json={
                "readings": readings,
                "ts": datetime.now().strftime("%H:%M:%S"),
            }).json()
        except Exception as e:
            print(f"  server unreachable: {e}")
            time.sleep(args.interval)
            continue

        n += 1
        if "captured" in r:
            print(f"  #{n}  calibrating  {r['captured']}/{r['target']} "
                  f"· {len(readings)} APs")
        elif "error" in r:
            print(f"  #{n}  {r['error']}")
        else:
            flag = "  BREACH" if r.get("breach") else ""
            print(f"  #{n}  ({r['x']:.2f}, {r['y']:.2f})  "
                  f"{r.get('zone') or '—'}  conf {r['conf']:.0%}{flag}")

        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n  stopped.")
