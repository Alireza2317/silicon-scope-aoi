"""The main application class for the Silicon-Scope AOI tool."""

import asyncio

from textual.app import App
from textual.binding import Binding

from src.core.inference import DetectionResult, InferenceEngine
from src.ui.screens import MainScreen


class SiliconScopeApp(App[None]):
	"""A Textual app for real-time object detection monitoring."""

	CSS_PATH = "app.tcss"
	BINDINGS = [
		Binding(key="q", action="quit", description="Quit"),
	]

	def __init__(
		self,
		output_queue: asyncio.Queue[list[DetectionResult] | str],
		inference_engine: InferenceEngine,
	) -> None:
		"""Initializes the application."""
		super().__init__()
		self.results_queue = output_queue
		self.inference_engine = inference_engine

		# screens
		self.main_screen = MainScreen()

	def on_mount(self) -> None:
		"""Called when the app is first mounted."""
		self.inference_engine.start()
		self.push_screen(self.main_screen)
		self.set_interval(1 / 30, self.process_results)

	def on_unmount(self) -> None:
		"""Called when the app is unmounted to clean up resources."""
		self.inference_engine.stop()

	async def process_results(self) -> None:
		"""Fetches results from the queue and updates the UI."""
		try:
			result = self.results_queue.get_nowait()
			video_feed = self.main_screen.video_feed
			log_palette = self.main_screen.log_palette


			# Update the log palette with either the error or the detections
			log_palette.update_log(result)

			# If there was a list of detections, update the video feed placeholder
			if isinstance(result, list):
				video_feed.update_feed(result)

		except asyncio.QueueEmpty:
			pass  # No new results, do nothing.
