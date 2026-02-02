"""The main application class for the Silicon-Scope AOI tool."""

import asyncio
import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque

from textual.app import App
from textual.binding import Binding

from src.core.inference import DetectionResult, Frame, InferenceEngine
from src.io.snapshot import save_detection_snapshot
from src.tui.screens import MainScreen
from src.tui.video_window import VideoWindow


class SiliconScopeApp(App[None]):
	"""A Textual app for real-time object detection monitoring."""

	CSS_PATH = "app.tcss"
	BINDINGS = [
		Binding(key="s", action="save_snapshot", description="Save Snapshot"),
		Binding(key="space", action="toggle_pause", description="Pause/Resume"),
		Binding(key="ctrl+l", action="toggle_log", description="Toggle Log"),
		Binding(key="q", action="quit", description="Quit"),
	]

	def __init__(
		self,
		output_queue: asyncio.Queue[tuple[Frame, list[DetectionResult]] | str],
		inference_engine: InferenceEngine,
		resume_event: threading.Event,
	) -> None:
		"""Initializes the application."""
		super().__init__()
		self.results_queue = output_queue
		self.inference_engine = inference_engine

		# Control Events
		self.resume_event = resume_event
		self.is_paused = False

		# UI Components
		self.main_screen = MainScreen()
		self.video_window = VideoWindow()
		self.fps_tracker: Deque[float] = deque(maxlen=20)

		# To keep track of the latest frame and detections for snapshotting
		self.latest_frame: Frame | None = None
		self.latest_detections: list[DetectionResult] | None = None

	def on_mount(self) -> None:
		"""Called when the app is first mounted."""
		self.inference_engine.start()
		self.push_screen(self.main_screen)
		self.set_interval(1 / 30, self.process_results)
		self.set_interval(1.0, self.update_fps)

	def on_unmount(self) -> None:
		"""Called when the app is unmounted to clean up resources."""
		self.inference_engine.stop()

		# setting the resume event to let the thread continue running if paused
		self.resume_event.set()

		if not self.video_window.is_closed():
			self.video_window.close()

	def action_save_snapshot(self) -> None:
		"""Saves the latest frame and detection data to a file."""
		if self.latest_frame is None:
			self.main_screen.log_widget.write(
				"[bold red]ERROR[/]: No frame available to save."
			)
			return

		try:
			output_dir = Path("snapshots")
			if self.latest_detections is None:
				raise ValueError("No detections available to save.")

			saved_path = save_detection_snapshot(
				self.latest_frame, self.latest_detections, output_dir
			)
			self.main_screen.log_widget.write(
				f"[yellow]Snapshot saved to {saved_path}.[jpg|json][/]\n"
			)
		except Exception as e:
			self.main_screen.log_widget.write(
				f"[bold red]ERROR[/]: Failed to save snapshot: {e}"
			)

	def action_toggle_pause(self) -> None:
		"""Toggles the paused state of the camera feed."""
		self.is_paused = not self.is_paused
		self.resume_event.clear() if self.is_paused else self.resume_event.set()

	def update_fps(self) -> None:
		"""Calculates and updates the FPS display."""
		if not self.fps_tracker:
			return

		times = list(self.fps_tracker)
		if len(times) < 2:
			return

		delta = times[-1] - times[0]
		if delta > 0:
			fps = (len(times) - 1) / delta
			self.main_screen.fps_widget.update_fps(fps)

	async def process_results(self) -> None:
		"""Fetches results from the queue and updates the UI."""
		try:
			result = self.results_queue.get_nowait()

			if isinstance(result, str):
				# It's an error or startup message
				if result.startswith("STARTUP::"):
					self.main_screen.log_widget.write(
						f"[yellow]Startup Info:[/]\n{result[9:]}\n"
					)
				else:
					self.main_screen.log_widget.write(f"[bold red]ERROR[/]: {result}\n")
			else:
				# It's a frame and its detections
				frame, detections = result
				self.fps_tracker.append(time.monotonic())

				# Update high-res GUI feed
				self.video_window.update(frame, detections)

				# If the user closes the high-res window, exit the app
				if self.video_window.is_closed():
					self.exit()

				# Update TUI dashboard
				self.main_screen.stats_widget.update_stats(detections)
				if detections:
					log_line = (
						f"Detected {len(detections)} objects: "
						+ ", ".join(
							f"{d.class_name} ({d.confidence:.2f})" for d in detections
						)
						+ "\n"
					)
					if not hasattr(self, "_last_log_line"):
						self._last_log_line = ""

					if log_line != self._last_log_line:
						self.main_screen.log_widget.write(log_line)
						self._last_log_line = log_line

				# Keep the latest frame and detections for snapshotting
				self.latest_frame = frame
				self.latest_detections = detections

		except asyncio.QueueEmpty:
			pass  # No new results, do nothing.
