"""The main application class for the Silicon-Scope AOI tool."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from textual.app import App
from textual.binding import Binding

if TYPE_CHECKING:
	from src.core.inference import InferenceEngine, QueueItem
from src.ui.screens import MainScreen
from src.ui.video_window import VideoWindow


class SiliconScopeApp(App[None]):
	"""A Textual app for real-time object detection monitoring."""

	CSS_PATH = "app.tcss"
	BINDINGS = [
		Binding(key="q", action="quit", description="Quit"),
	]

	def __init__(
		self,
		output_queue: asyncio.Queue[QueueItem],
		inference_engine: InferenceEngine,
	) -> None:
		"""Initializes the application."""
		super().__init__()
		self.results_queue = output_queue
		self.inference_engine = inference_engine

		# UI Components
		self.main_screen = MainScreen()
		self.video_window = VideoWindow()

	def on_mount(self) -> None:
		"""Called when the app is first mounted."""
		self.inference_engine.start()
		self.push_screen(self.main_screen)

		self.set_interval(1 / 30, self.process_results)

	def on_unmount(self) -> None:
		"""Called when the app is unmounted to clean up resources."""
		self.inference_engine.stop()
		self.video_window.close()

	async def process_results(self) -> None:
		"""Fetches results from the queue and updates the UI."""
		try:
			result: QueueItem = self.results_queue.get_nowait()

			if isinstance(result, str):
				# It's an error message
				self.main_screen.log_palette.update_log(result)
			else:
				# It's a frame and its detections
				frame, detections = result
				self.main_screen.log_palette.update_log(detections)

				self.video_window.update(frame, detections)

		except asyncio.QueueEmpty:
			pass  # No new results, do nothing.
