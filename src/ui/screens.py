"""Screens for the SiliconScope AOI application."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Footer, Header, Static

if TYPE_CHECKING:
	from src.core.inference import DetectionResult


class VideoFeed(Static):
	"""A placeholder widget for the video feed."""

	def update_feed(self, detections: list[DetectionResult]) -> None:
		"""Update the video feed with new detections."""
		self.update(f"Video Feed Placeholder\nDetections: {len(detections)}")


class LogPalette(Static):
	"""A placeholder widget for logging output."""

	def update_log(self, result: list[DetectionResult] | str) -> None:
		"""Update the log with new results or an error message."""

		if isinstance(result, str):
			self.update(f"[bold red]ERROR[/]: {result}")
		elif isinstance(result, list):
			summary: str = f"Detections: {len(result)}\n"
			summary += "\n".join(
				f"  - {res.class_name} (Conf: {res.confidence:.2f})" for res in result
			)
			self.update(summary)


class MainScreen(Screen[None]):
	"""The main screen for the application.

	It contains the video feed and a log palette.
	"""

	def __init__(self, *args: Any, **kwargs: Any) -> None:
		super().__init__(*args, **kwargs)

		self.header = Header()
		self.footer = Footer()
		self.video_feed = VideoFeed("Video Feed Placeholder", id="video-feed")
		self.log_palette = LogPalette("Log Palette Placeholder", id="log-view")

	def compose(self) -> ComposeResult:
		"""Compose the layout of the main screen."""
		yield Vertical(
			self.header,
			self.video_feed,
			self.log_palette,
			self.footer,
			id="main-layout",
		)
