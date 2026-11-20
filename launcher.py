"""Roundel 3D launcher.

Starts the bundled Streamlit app on a free loopback port, waits until the
server is actually answering, then opens the browser at it.
"""

from pathlib import Path
import os
import socket
import sys
import threading
import time
import webbrowser

from streamlit.web import cli as stcli


def bundled_path(*parts: str) -> str:
    """A read-only file shipped inside the bundle."""
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return str(base.joinpath(*parts))


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def open_when_ready(port: int, url: str, timeout: float = 120.0) -> None:
    """Poll the port and only open the browser once Streamlit answers.

    A frozen build has to unpack torch/skimage/monai before the server
    binds, which can take far longer than a fixed delay. Opening early
    gives the user a connection-refused page on a working app.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                break
        except OSError:
            time.sleep(0.5)
    else:
        print(f"Server did not start within {timeout:.0f}s. Open {url} manually.")
        return
    try:
        webbrowser.open_new_tab(url)
    except Exception:
        print(f"Could not open a browser. Go to {url}")


def main() -> None:
    app_path = bundled_path("roundel_app.py")
    port = free_port()
    url = f"http://127.0.0.1:{port}"

    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
    # The source watcher has nothing useful to watch in a frozen bundle and
    # can trip over the _internal tree.
    os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

    print(f"Roundel 3D starting on {url}")
    threading.Thread(target=open_when_ready, args=(port, url), daemon=True).start()

    sys.argv = [
        "streamlit", "run", app_path,
        "--server.address=127.0.0.1",
        f"--server.port={port}",
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
        "--global.developmentMode=false",
        "--server.enableXsrfProtection=false",
        "--server.enableCORS=false",
        # .streamlit/config.toml is bundled into _internal\, where Streamlit
        # never looks for it (only $CWD/.streamlit and ~/.streamlit), so the
        # message size has to be set here or large scans fail to upload.
        "--server.maxMessageSize=600",
        # Hides the Deploy button and the developer menu items.
        "--client.toolbarMode=minimal",
    ]
    raise SystemExit(stcli.main())


if __name__ == "__main__":
    main()