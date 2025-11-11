"""
Utility functions for Streamlit app: progress tracking, game execution, and file management.
"""
import os
import sys
import threading
import queue
import time
from pathlib import Path
from typing import Callable, Dict, Any, Optional
import streamlit as st

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class ProgressTracker:
    """Thread-safe progress tracker for long-running game simulations."""
    
    def __init__(self, total: int, desc: str = "Progress"):
        self.total = total
        self.current = 0
        self.desc = desc
        self.lock = threading.Lock()
        self.start_time = time.time()
        self.messages = []
    
    def update(self, n: int = 1, message: str = ""):
        """Update progress by n steps."""
        with self.lock:
            self.current = min(self.current + n, self.total)
            if message:
                self.messages.append(message)
    
    def get_progress(self) -> tuple:
        """Get current progress as (current, total, percentage, elapsed_time)."""
        with self.lock:
            elapsed = time.time() - self.start_time
            percentage = (self.current / self.total * 100) if self.total > 0 else 0
            return self.current, self.total, percentage, elapsed
    
    def get_messages(self) -> list:
        """Get all progress messages."""
        with self.lock:
            return self.messages.copy()


def run_game_with_progress(
    game_func: Callable,
    progress_key: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Run a game function with progress tracking.
    
    Args:
        game_func: Function that runs the game simulation
        progress_key: Unique key for Streamlit progress tracking
        **kwargs: Arguments to pass to game_func
    
    Returns:
        Dictionary with results and metadata
    """
    progress_bar = st.progress(0, text="Initializing...")
    status_text = st.empty()
    
    result_queue = queue.Queue()
    error_queue = queue.Queue()
    
    def run_in_thread():
        try:
            result = game_func(**kwargs)
            result_queue.put(result)
        except Exception as e:
            error_queue.put(e)
    
    thread = threading.Thread(target=run_in_thread, daemon=True)
    thread.start()
    
    # Poll for progress updates
    last_update = 0
    while thread.is_alive():
        time.sleep(0.1)  # Update every 100ms
        
        # Check for errors
        if not error_queue.empty():
            error = error_queue.get()
            progress_bar.empty()
            status_text.empty()
            st.error(f"Error: {str(error)}")
            return {"error": str(error)}
        
        # Update progress (simplified - actual progress tracking would need game-specific hooks)
        elapsed = time.time() - last_update
        if elapsed > 0.5:  # Update every 500ms
            progress_bar.progress(0.5, text="Running simulation...")
            last_update = time.time()
    
    thread.join(timeout=0.1)
    
    if not result_queue.empty():
        result = result_queue.get()
        progress_bar.progress(1.0, text="Complete!")
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        return result
    else:
        progress_bar.empty()
        status_text.empty()
        return {"error": "Simulation did not complete"}


def ensure_demo_dir(game_name: str) -> Path:
    """Ensure demo directory exists for a game."""
    demo_dir = Path("demos") / game_name
    demo_dir.mkdir(parents=True, exist_ok=True)
    return demo_dir


def list_demo_files(game_name: str, extension: str = ".gif") -> list:
    """List all demo files for a game."""
    demo_dir = Path("demos") / game_name
    if not demo_dir.exists():
        return []
    
    files = sorted(demo_dir.glob(f"*{extension}"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [f.name for f in files]


def get_latest_demo_file(game_name: str, extension: str = ".gif") -> Optional[Path]:
    """Get the most recent demo file for a game."""
    files = list_demo_files(game_name, extension)
    if not files:
        return None
    return Path("demos") / game_name / files[0]


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

