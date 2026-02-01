"""Provides a camera feed source for the application."""

import logging
from collections.abc import Generator

import cv2
import numpy as np


def generate_camera_frames() -> Generator[np.ndarray, None, None]:
	"""
	A generator function that yields frames from the default camera.

	It handles opening and releasing the camera resource. If the camera
	cannot be opened, it logs an error and exits gracefully.
	"""
	cap = cv2.VideoCapture(0)
	if not cap.isOpened():
		logging.error("ERROR: Cannot open camera.")
		return  # Stop the generator if the camera is not available

	try:
		while True:
			ret, frame = cap.read()
			if not ret:
				logging.error("ERROR: Can't receive frame (stream end?).")
				break
			yield frame
	finally:
		cap.release()
		logging.info("Camera feed stopped and released.")
