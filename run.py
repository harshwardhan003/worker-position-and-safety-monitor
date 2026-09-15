"""
PhantomMap 5.0 — Just run this.
py -3.11 run.py
Then on phone: python scanner.py --host 192.168.1.10 --mode motion
"""
import sys, time, threading, subprocess, webbrowser

print("""
╔══════════════════════════════════════════════════════════╗
║          📡  PhantomMap 5.0                              ║
║  Walk with your phone — dot moves with you               ║
║  No setup · No zones · No calibration                    ║
╠══════════════════════════════════════════════════════════╣
║  Dashboard → http://localhost:8501                       ║
╚══════════════════════════════════════════════════════════╝
""")

exe = sys.executable

for pkg in ["flask-cors"]:
    try: __import__(pkg.replace("-","_"))
    except ImportError:
        subprocess.run([exe,"-m","pip","install",pkg,"-q"])

threading.Thread(
    target=lambda: subprocess.run([exe,"server.py"]),
    daemon=True
).start()
print("[Run] Server started on :5050")
time.sleep(2)

subprocess.Popen([exe,"dashboard.py"])
print("[Run] Dashboard started on :8501")
time.sleep(2)

webbrowser.open("http://localhost:8501")
print("[Run] Browser opened")
print("\nOn your phone in Termux:")
print("  python scanner.py --host 192.168.1.10 --mode motion\n")
print("Click anywhere on the map to set your start position.")
print("Press Ctrl+C to stop.\n")

try:
    while True: time.sleep(1)
except KeyboardInterrupt:
    print("\nStopped.")
