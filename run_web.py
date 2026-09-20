"""Football Prediction System - Flask Run Script."""
import sys
import os
import time
import codecs
import subprocess
import signal

if sys.stdout.encoding != "utf-8":
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "replace")

def _kill_old_server(port=5000):
    """Mevcut porttaki tum baglantilari (dinleyen + kurulu) tutan islemleri zorla oldur."""
    try:
        import ctypes
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             f"Get-NetTCPConnection -LocalPort {port} -ErrorAction SilentlyContinue | "
             f"Select-Object -ExpandProperty OwningProcess -Unique"],
            capture_output=True, text=True, timeout=5
        )
        pids = [p.strip() for p in result.stdout.strip().split('\n') if p.strip()]
        my_pid = os.getpid()
        for pid in pids:
            if not pid:
                continue
            try:
                if int(pid) == my_pid or int(pid) == 0 or int(pid) == 4:
                    continue
            except ValueError:
                continue
            try:
                os.kill(int(pid), signal.SIGTERM)
                print(f"  [kill] Eski surec PID {pid} durduruldu (port {port})", flush=True)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                Handle = ctypes.windll.kernel32.OpenProcess(1, False, int(pid))
                if Handle:
                    ctypes.windll.kernel32.TerminateProcess(Handle, 1)
                    ctypes.windll.kernel32.CloseHandle(Handle)
            except Exception:
                pass
        time.sleep(1)
    except Exception:
        pass

import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

from web.app import app, _load_model, _start_auto_fetch

if __name__ == "__main__":
    _kill_old_server(5000)

    print("Football Prediction System starting...", flush=True)
    print("http://localhost:5000", flush=True)
    _load_model()
    print("Press Ctrl+C to stop", flush=True)

    try:
        from jobs.daily_sync import run_schedule
        _sync_thread = threading.Thread(target=run_schedule, daemon=True)
        _sync_thread.start()
        print("Gunluk veri guncelleme scheduler kuruldu (her 24h)", flush=True)
    except Exception as e:
        print(f"Gunluk sync baslatilamadi: {e}", flush=True)

    # Otomatik sonuc cekme baslat (her 2 dk)
    _start_auto_fetch()

    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
