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
		self.window_name = window_name
		self._is_created = False

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

		cv2.imshow(self.window_name, display_frame)
		self._is_created = True  # Mark the window as created
		cv2.waitKey(1)  # Necessary for the window to update

	def is_closed(self) -> bool:
		"""
		Checks if the video window has been closed by the user.

		Returns:
			True if the window was closed, False otherwise.
		"""
		if not self._is_created:
			return False
		try:
			# A visible window will have a property > 0. If it's closed, this will be < 1.
			return cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1
		except cv2.error:
			# This can happen if the window is forcefully closed.
			return True

	def close(self) -> None:
		"""Closes the video window if it exists."""
		if self._is_created:
			cv2.destroyWindow(self.window_name)
