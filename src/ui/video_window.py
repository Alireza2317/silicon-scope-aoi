"""
This module handles the high-resolution video feed window using OpenCV.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2

if TYPE_CHECKING:
	from src.core.inference import DetectionResult, Frame


class VideoWindow:
	"""Manages a high-resolution window for displaying the video feed."""

	def __init__(self, window_name: str = "High-Resolution Feed") -> None:
		"""Initializes the video window."""
		self.name = window_name

	def update(self, frame: Frame, detections: list[DetectionResult]) -> None:
		"""
		Updates the window with a new frame and draws the detections.

		Args:
			frame: The new video frame to display.
			detections: A list of detections to draw on the frame.
		"""
		# Make a copy to avoid drawing on the original frame
		display_frame = frame.copy()

		for detection in detections:
			x1, y1, x2, y2 = map(int, detection.box)
			label = f"{detection.class_name} ({detection.confidence:.2f})"
			cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
			cv2.putText(
				display_frame,
				label,
				(x1, y1 - 10),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.5,
				(0, 255, 0),
				2,
			)

		cv2.imshow(self.name, display_frame)
		cv2.waitKey(1)  # Necessary for the window to update

	def close(self) -> None:
		"""Closes the video window."""
		cv2.destroyWindow(self.name)
