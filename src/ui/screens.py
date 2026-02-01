"""Screens for the SiliconScope AOI application."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cv2
from PIL import Image
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import Footer, Header, Static

if TYPE_CHECKING:
	from src.core.inference import DetectionResult, Frame


def render_frame_as_ansi(frame: Frame, widget: Widget) -> str:
	"""
	Converts a numpy array frame to a string of ANSI color codes using
	half-block characters for better fidelity.
	"""
	# The frame from cv2 is BGR, so we convert it to RGB for Pillow
	frame_rgb: Frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	img = Image.fromarray(frame_rgb)

	widget_width = widget.content_size.width
	# Adjust for character aspect ratio. The height is now effectively doubled
	# because we are drawing two pixel rows per character row.
	widget_height_adjusted = widget.content_size.height * 2
	if widget_width <= 0 or widget_height_adjusted <= 0:
		return ""

	img.thumbnail((widget_width, widget_height_adjusted), Image.Resampling.LANCZOS)
	img_width, img_height = img.size

	pad_x = (widget_width - img_width) // 2
	pad_y = (widget.content_size.height - (img_height // 2)) // 2
	left_padding = " " * pad_x

	buffer = []
	buffer.extend(["\n"] * pad_y)

	# Iterate over the image height in steps of 2
	for y in range(0, img_height, 2):
		buffer.append(left_padding)
		for x in range(img_width):
			# Top pixel (foreground color)
			top_pixel = img.getpixel((x, y))
			if not isinstance(top_pixel, tuple):
				continue

			r1, g1, b1 = top_pixel

			# # Bottom pixel (background color)
			if y + 1 < img_height:
				bottom_pixel = img.getpixel((x, y + 1))
				if not isinstance(bottom_pixel, tuple):
					continue
				r2, g2, b2 = bottom_pixel
			else:
				# If there's no bottom pixel, use black
				r2, g2, b2 = 0, 0, 0

			# Set foreground for top pixel, background for bottom pixel
			buffer.append(f"\x1b[38;2;{r1};{g1};{b1}m\x1b[48;2;{r2};{g2};{b2}m▄")

		buffer.append("\x1b[0m\n")  # Reset colors and add newline

	bottom_pad_count = widget.content_size.height - (img_height // 2) - pad_y
	buffer.extend(["\n"] * max(0, bottom_pad_count))

	return "".join(buffer)


class VideoFeed(Static):
	"""A widget to display video frames with bounding boxes."""

	def update_feed(self, frame: Frame, detections: list[DetectionResult]) -> None:
		"""Draw detections on the frame and render it as ANSI art."""
		# Draw bounding boxes and labels on the frame
		for detection in detections:
			x1, y1, x2, y2 = map(int, detection.box)
			label = f"{detection.class_name} ({detection.confidence:.2f})"
			cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
			cv2.putText(
				frame,
				label,
				(x1, y1 - 10),
				fontFace=cv2.FONT_HERSHEY_SIMPLEX,
				fontScale=0.5,
				color=(0, 255, 0),  # green
				thickness=1,
			)

		# Convert the frame with drawings to an ANSI string
		ansi_frame_str = render_frame_as_ansi(frame, self)
		# Create a Rich Text object from the ANSI string and update the widget
		self.update(Text.from_ansi(ansi_frame_str))


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
