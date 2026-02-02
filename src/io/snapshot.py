"""This module handles I/O operations, such as saving detection snapshots."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
	from src.core.inference import DetectionResult


def save_detection_snapshot(
	frame: np.ndarray, detections: list[DetectionResult], output_dir: Path
) -> Path:
	"""
	Saves the video frame as an image and the detection data as a JSON file.

	A timestamp is used to create a unique filename for the pair of files.

	Args:
		frame: The video frame (as a NumPy array) to save.
		detections: The list of detection data to save.
		output_dir: The directory where the snapshot files will be saved.

	Returns:
		The base path of the saved snapshot (without the file extension).
	"""
	output_dir.mkdir(parents=True, exist_ok=True)
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
	base_path = output_dir / timestamp

	# Create a copy to draw boxes on
	annotated_frame = frame.copy()
	for detection in detections:
		x1, y1, x2, y2 = map(int, detection.box)
		label = f"{detection.class_name} ({detection.confidence:.2f})"
		# Draw bounding box in red (BGR: 0, 0, 255)
		cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
		cv2.putText(
			annotated_frame,
			label,
			(x1, y1 - 10),
			cv2.FONT_HERSHEY_DUPLEX,
			0.9,
			(0, 0, 200),
			2,
		)

	# Save the image file
	image_path = base_path.with_suffix(".jpg")
	cv2.imwrite(str(image_path), annotated_frame)

	# Save the detection data as a JSON file
	json_path = base_path.with_suffix(".json")
	detection_data = [d.model_dump() for d in detections]
	with json_path.open("w") as f:
		json.dump(detection_data, f, indent=4)

	return base_path
