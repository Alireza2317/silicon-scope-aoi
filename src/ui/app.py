"""The main application class for the Silicon-Scope AOI tool."""

import asyncio
import threading
import time
from collections import deque
from typing import Deque

from textual.app import App
from textual.binding import Binding

from src.core.inference import DetectionResult, Frame, InferenceEngine
from src.ui.screens import MainScreen
from src.ui.video_window import VideoWindow


class SiliconScopeApp(App[None]):
	"""A Textual app for real-time object detection monitoring."""

	CSS_PATH = "app.tcss"
	BINDINGS = [
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

	def on_mount(self) -> None:
		"""Called when the app is first mounted."""
		self.inference_engine.start()
		self.push_screen(self.main_screen)
		self.set_interval(1 / 30, self.process_results)
		self.set_interval(1.0, self.update_fps)

	def on_unmount(self) -> None:
		"""Called when the app is unmounted to clean up resources."""
		self.inference_engine.stop()
		self.video_window.close()

		# setting the resume event to let the thread continue running if paused
		self.resume_event.set()

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
				# It's an error message
				self.main_screen.log_widget.write(f"[bold red]ERROR[/]: {result}")
			else:
				# It's a frame and its detections
				frame, detections = result
				self.fps_tracker.append(time.monotonic())

				# Update high-res GUI feed
				self.video_window.update(frame, detections)

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

		except asyncio.QueueEmpty:
			pass  # No new results, do nothing.
