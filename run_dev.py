"""
Convenience launcher: runs the FastAPI backend and Streamlit frontend together for
local development, so you don't need two terminals. Ctrl+C stops both.
"""

import subprocess
import sys
import time


def main():
    backend = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--reload", "--port", "8000"]
    )
    frontend = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "streamlit_app.py"]
    )
    procs = [backend, frontend]

    try:
        while all(p.poll() is None for p in procs):
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            p.wait()


if __name__ == "__main__":
    main()
