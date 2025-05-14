# auto_reload.py
import time
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

PYQT_FILE = "Main.py"

class ReloadHandler(FileSystemEventHandler):
    def __init__(self):
        self.process = None
        self.start_app()

    def start_app(self):
        print("🚀 Starting PyQt App...")
        self.process = subprocess.Popen(["python", PYQT_FILE])

    def restart_app(self):
        print("🔁 Reload detected. Restarting...")
        self.process.kill()
        self.start_app()

    def on_modified(self, event):
        if event.src_path.endswith(".py"):
            self.restart_app()

if __name__ == "__main__":
    event_handler = ReloadHandler()
    observer = Observer()
    observer.schedule(event_handler, path=".", recursive=True)
    observer.start()

    print("👀 Watching for changes... Press Ctrl+C to stop.")
    print("👀 Watching for changes... Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("🛑 Stopping...")
        observer.stop()
        event_handler.process.kill()
    observer.join()
