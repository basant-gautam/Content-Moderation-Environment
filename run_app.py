from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


def _base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    return Path(__file__).resolve().parent


def _wait_for_backend(url: str, timeout_seconds: int = 20) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return True
        except (URLError, OSError):
            pass
        time.sleep(0.4)
    return False


def main() -> None:
    base_dir = _base_dir()
    env = os.environ.copy()

    backend_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "server.app:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ]

    backend_process = subprocess.Popen(backend_cmd, cwd=str(base_dir), env=env)

    try:
        if _wait_for_backend("http://127.0.0.1:8000/health"):
            frontend_path = base_dir / "frontend" / "index.html"
            if frontend_path.exists():
                webbrowser.open(frontend_path.resolve().as_uri())
            else:
                webbrowser.open("http://127.0.0.1:8000/docs")
            print("Backend started at http://127.0.0.1:8000")
            print("Frontend opened in your browser.")
        else:
            print("Backend did not become ready in time. Opening API docs instead.")
            webbrowser.open("http://127.0.0.1:8000/docs")

        print("Press Ctrl+C to stop the backend.")
        backend_process.wait()
    except KeyboardInterrupt:
        print("Stopping backend...")
    finally:
        if backend_process.poll() is None:
            if os.name == "nt":
                backend_process.send_signal(signal.CTRL_BREAK_EVENT)
                time.sleep(0.5)
            backend_process.terminate()
            try:
                backend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                backend_process.kill()


if __name__ == "__main__":
    main()
