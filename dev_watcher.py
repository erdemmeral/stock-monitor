import subprocess
import time
import logging
import os
import sys
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class CodeChangeHandler(FileSystemEventHandler):
    def __init__(self):
        self.process = None
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def start_process(self):
        # Terminate existing process if running
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()

        # Start new process
        self.logger.info("🔄 Restarting Stock Monitor...")
        self.process = subprocess.Popen([
            sys.executable,  # Use the same Python interpreter
            "-m", "app.monitoring.real_time_monitor"
        ])
        self.logger.info("✅ Stock Monitor restarted successfully")

    def on_modified(self, event):
        # Only reload on Python file changes
        if event.src_path.endswith('.py'):
            # Ignore changes in certain directories
            ignore_dirs = ['/venv', '/.git', '/__pycache__']
            if not any(ignore_dir in event.src_path for ignore_dir in ignore_dirs):
                self.logger.info(f"🔍 Detected change in: {event.src_path}")
                self.start_process()

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Initialize file change handler
    event_handler = CodeChangeHandler()
    event_handler.start_process()  # Initial start of the application

    # Setup file system observer
    observer = Observer()
    observer.schedule(event_handler, path='.', recursive=True)
    observer.start()

    logger.info("🚀 Development Watcher Started. Monitoring for code changes...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        logger.info("🛑 Development Watcher Stopped")
    
    observer.join()

if __name__ == "__main__":
    main()