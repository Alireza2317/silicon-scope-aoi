"""The main entry point for the application."""

import argparse
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


def feed_worker(
	frame_generator: Generator[np.ndarray, None, None],
	engine: InferenceEngine,
	stop_event: threading.Event,
	resume_event: threading.Event,
) -> None:
	"""
	Consumes frames from a generator and submits them to the inference engine.
	"""
	engine.started_event.wait()  # Wait for the engine to be ready

	for frame in frame_generator:
		if stop_event.is_set():
			break

		# If the resume event is not set(cleared), wait until it is set.
		resume_event.wait()

		engine.submit_frame(frame)

		time.sleep(0.01)


def main() -> None:
	"""The main entry point for the application."""
	parser = argparse.ArgumentParser(description="Silicon Scope AOI Application")
	parser.add_argument(
		"--ui",
		type=str,
		default="gui",
		choices=["gui", "tui"],
		help="The user interface to run ('gui' or 'tui').",
	)
	args = parser.parse_args()

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
	pause_event = threading.Event()

	# starting in the running state
	pause_event.set()

	inference_engine = InferenceEngine(inference_config, inference_queue)

	# The camera generator runs in a background thread
	camera_generator = generate_camera_frames()
	feed_thread = threading.Thread(
		target=feed_worker,
		args=(camera_generator, inference_engine, stop_event, pause_event),
		daemon=True,
	)
	feed_thread.start()

	if args.ui == "tui":
		from src.tui.app import SiliconScopeApp

		app = SiliconScopeApp(inference_queue, inference_engine, pause_event)
		app.run()
	else:
		# We will create the GUI app here in the next step
		logging.info("GUI mode selected (implementation pending).")
		print("GUI mode selected (implementation pending).")
		print("Press Ctrl+C to exit.")
		# Keep the main thread alive to allow the background thread to run
		# In a real GUI, the app's mainloop would do this.
		try:
			while not stop_event.is_set():
				feed_thread.join(0.1)
		except KeyboardInterrupt:
			print("Exiting...")

	# Signal the feed worker to stop and wait for it
	stop_event.set()
	feed_thread.join()


if __name__ == "__main__":
	main()
