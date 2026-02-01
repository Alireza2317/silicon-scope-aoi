"""
Core Inference Engine for Spec-Check.

This module contains the primary logic for loading the YOLOv8 model
and performing object detection in a separate, non-blocking thread.
"""

import asyncio
import threading
from pathlib import Path
from types import TracebackType
from typing import Self

import numpy as np
from pydantic import BaseModel, Field
from ultralytics.models import YOLO

type BoxCoordinate = tuple[float, float, float, float]
type Frame = np.ndarray
type QueueItem = tuple[Frame, list[DetectionResult]] | str


class DetectionResult(BaseModel):
	"""
	Represents a single object detection made by the YOLO model.

	This is a data-transfer object (DTO) used to pass structured results
	from the inference thread to the UI thread.

	Attributes:
	    box: Bounding box coordinates in (x1, y1, x2, y2) format(BoxCoordinate).
	    class_id: The integer ID of the detected class.
	    class_name: The string name of the detected class.
	    confidence: The confidence score of the detection (0.0 to 1.0).
	"""

	box: BoxCoordinate = Field(
		..., description="Bounding box coordinates (x1, y1, x2, y2)"
	)
	class_id: int = Field(..., description="The ID of the detected class")
	class_name: str = Field(..., description="The name of the detected class")
	confidence: float = Field(..., description="The confidence score of the detection")


class InferenceConfig(BaseModel):
	"""
	Configuration settings for the InferenceEngine.

	Provides a type-safe way to manage engine parameters.

	Attributes:
	    model_path: The file path to the YOLOv8 model weights (.pt file).
	    confidence_threshold: The minimum confidence score to consider a detection valid.
	    image_size: The square dimension (e.g., 640) to which input images are resized.
	"""

	model_path: Path = Field(
		Path("models/yolov8n.pt"), description="Path to the YOLOv8 model weights"
	)
	confidence_threshold: float = Field(
		0.5, ge=0.0, le=1.0, description="Minimum confidence for detection"
	)
	image_size: int = Field(
		320, gt=0, description="The size for inference input images"
	)


class InferenceEngine:
	"""
	Performs real-time object detection in a dedicated background thread.

	This class acts as a context manager to manage the lifecycle of its
	internal inference thread. It decouples AI inference from the main UI
	thread, loads a YOLOv8 model, processes image frames, and places results
	in a thread-safe queue for consumption by other parts of the application.
	"""

	def __init__(
		self,
		config: InferenceConfig,
		output_queue: asyncio.Queue[QueueItem],
	) -> None:
		"""
		Initializes the InferenceEngine with its configuration and output queue.

		Args:
			config: Configuration object with model path and other settings.
			output_queue: A thread-safe queue to send results or errors to the UI.
		"""
		self.config = config
		self._output_queue = output_queue
		self._loop: asyncio.AbstractEventLoop | None = None
		self._stop_event = threading.Event()
		self.started_event = threading.Event()
		self._model: YOLO | None = None
		self._frame_to_process: np.ndarray | None = None
		self._frame_lock = threading.Lock()
		self._thread: threading.Thread | None = None

	def __enter__(self) -> Self:
		"""
		Context manager entry point. Starts the background inference thread.

		Returns:
		    The InferenceEngine instance itself.
		"""
		self.start()

		return self

	def __exit__(
		self,
		exc_type: type[BaseException] | None,
		exc_val: BaseException | None,
		exc_tb: TracebackType | None,
	) -> None:
		"""
		Signals the inference thread to stop and waits for its termination.
		"""
		self.stop()

	def start(self) -> None:
		"""
		Starts the background inference thread.

		It captures the running asyncio event loop from the calling thread (expected
		to be the main UI thread). This loop is used to safely dispatch inference
		results back to the async UI components.
		"""
		if self._thread is not None and self._thread.is_alive():
			return  # Thread is already running

		try:
			self._loop = asyncio.get_running_loop()
		except RuntimeError:
			raise RuntimeError(
				f"{self.__class__.__name__}.start() must be called from a running"
				+ " asyncio event loop."
			)

		self._thread = threading.Thread(target=self._run_loop, daemon=True)
		self._thread.start()
		self.started_event.set()

	def stop(self) -> None:
		"""Signals the inference thread to stop gracefully and waits for it to finish."""
		self._stop_event.set()
		if self._thread is not None:
			self._thread.join()

	def _run_loop(self) -> None:
		"""
		The main loop for the background inference thread.

		This method runs in a separate thread. It loads the model and then enters a
		loop, continuously processing the most recent frame. It uses the captured
		event loop from the main thread to safely put results into the asyncio queue.
		"""
		if not self._loop:
			# This should not be reachable if start() is used correctly.
			return

		try:
			self._model = YOLO(self.config.model_path)
			# Perform a dummy prediction to initialize the model
			# and avoid a long delay on the first frame.
			dummy_img = np.zeros(
				(self.config.image_size, self.config.image_size, 3), dtype=np.uint8
			)
			self._model(dummy_img, imgsz=self.config.image_size, verbose=False)
		except Exception as e:
			error_message: str = f"ERROR: Failed to load model: {e}"
			self._loop.call_soon_threadsafe(
				self._output_queue.put_nowait, error_message
			)
			return

		while not self._stop_event.is_set():
			frame: Frame | None = None
			with self._frame_lock:
				if self._frame_to_process is not None:
					frame = self._frame_to_process
					self._frame_to_process = None  # Consume the frame

			if frame is not None:
				try:
					results: list[DetectionResult] = self._process_frame(frame)
					self._loop.call_soon_threadsafe(
						self._output_queue.put_nowait, (frame, results)
					)
				except Exception as e:
					error_message = f"ERROR: Failed to process frame: {e}"
					self._loop.call_soon_threadsafe(
						self._output_queue.put_nowait, error_message
					)
			else:
				# Wait briefly to prevent busy-waiting when no frames are available.
				self._stop_event.wait(0.01)

	def _process_frame(self, frame: np.ndarray) -> list[DetectionResult]:
		"""
		Processes a single image frame to perform object detection.

		Args:
			frame: The input image frame in NumPy array format.

		Returns:
			A list of DetectionResult objects for all valid detections.
		"""
		if self._model is None:
			return []

		# The result from the model is an iterable of result objects
		results = self._model(
			frame,
			imgsz=self.config.image_size,
			conf=self.config.confidence_threshold,
			verbose=False,
		)

		detections: list[DetectionResult] = []
		for res in results:
			# res.boxes.data is a tensor of [x1, y1, x2, y2, conf, cls]
			for row in res.boxes.data.tolist():
				if len(row) != 6:
					continue  # Skip malformed results

				(x1, y1, x2, y2, confidence, class_id_float) = row
				class_id: int = int(class_id_float)

				detections.append(
					DetectionResult(
						box=(x1, y1, x2, y2),
						class_id=class_id,
						class_name=self._model.names.get(class_id, "Unknown"),
						confidence=confidence,
					)
				)
		return detections

	def submit_frame(self, frame: np.ndarray) -> None:
		"""
		Submits a new frame for processing. This is thread-safe.

		Args:
		    frame: The latest camera frame.
		"""
		with self._frame_lock:
			self._frame_to_process = frame

	def is_loop_running(self) -> bool:
		"""
		Checks if the internal event loop is running.

		Returns:
			True if the event loop is running, False otherwise.
		"""
		return (self._loop is not None) and self._loop.is_running()
