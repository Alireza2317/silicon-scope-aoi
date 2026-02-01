import asyncio
import logging
import threading
import time
from collections.abc import Generator

import numpy as np

from src.core.inference import (
	InferenceConfig,
	InferenceEngine,
	QueueItem,
)
from src.sources.camera import generate_camera_frames
from src.ui.app import SiliconScopeApp


def feed_worker(
	frame_generator: Generator[np.ndarray, None, None],
	engine: InferenceEngine,
	stop_event: threading.Event,
) -> None:
	"""
	Consumes frames from a generator and submits them to the inference engine.
	"""
	engine.started_event.wait()  # Wait for the engine to be ready

	for frame in frame_generator:
		if stop_event.is_set():
			break
		engine.submit_frame(frame)

		time.sleep(0.01)


def main() -> None:
	"""The main entry point for the application."""
	logging.basicConfig(
		level=logging.INFO,
		filename="debug.log",
		filemode="w",
		format="%(asctime)s - %(levelname)s - %(message)s",
	)
	# Use an asyncio queue for non-blocking communication with the UI.
	inference_queue: asyncio.Queue[QueueItem] = asyncio.Queue()
	inference_config = InferenceConfig()  # type: ignore
	stop_event = threading.Event()

	inference_engine = InferenceEngine(inference_config, inference_queue)
	app = SiliconScopeApp(inference_queue, inference_engine)

	# Create the frame generator and the worker thread.
	camera_generator = generate_camera_frames()
	feed_thread = threading.Thread(
		target=feed_worker,
		args=(camera_generator, inference_engine, stop_event),
		daemon=True,
	)
	feed_thread.start()

	app.run()

	# Signal the camera feed to stop and wait for it.
	stop_event.set()
	feed_thread.join()


if __name__ == "__main__":
	main()
