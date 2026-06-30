from pathlib import Path
import os, sys, socket, threading, webbrowser
from streamlit.web import cli as stcli

def bundled_path(*parts: str) -> str:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return str(base.joinpath(*parts))

def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]

def main():
    app_path = bundled_path("roundel_app_clinical.py")   # your app script
    port = free_port()
    url = f"http://127.0.0.1:{port}"

    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")

    threading.Timer(1.5, lambda: webbrowser.open_new_tab(url)).start()

    sys.argv = [
        "streamlit", "run", app_path,
        "--server.address=127.0.0.1",
        f"--server.port={port}",
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
        "--global.developmentMode=false",
        "--server.enableXsrfProtection=false",
        "--server.enableCORS=false",
    ]
    raise SystemExit(stcli.main())

if __name__ == "__main__":
    main()