import asyncio
import logging
import random
import threading
import time

import cv2
import numpy as np
from matplotlib import pyplot as plt

from src.core.inference import DetectionResult, InferenceConfig, InferenceEngine
from src.ui.app import SiliconScopeApp


def dummy_feed(
	engine: InferenceEngine,
	stop_event: threading.Event,
	engine_started_event: threading.Event,
) -> None:
	"""Generates and submits dummy frames to the inference engine."""
	# Wait for the inference engine to be started by the app.
	engine_started_event.wait()

	while not stop_event.is_set():
		try:
			# read dummy image as a np array
			frame = plt.imread(random.choice(('p.png', 'pic.png')))

			# Ensure the frame is in 3-channel, uint8 format
			if frame.ndim == 2:  # Grayscale
				frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
			elif frame.shape[2] == 4:  # RGBA
				frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

			if frame.dtype != np.uint8:
				frame = (frame * 255).astype(np.uint8)

			engine.submit_frame(frame)
		except FileNotFoundError:
			logging.exception(
				"ERROR: image not found. Make sure it is in the root directory."
			)
			stop_event.set()  # Stop the feed if the image is missing
		except Exception:
			logging.exception("ERROR in dummy_feed")

		time.sleep(1)


def main() -> None:
	"""The main entry point for the application."""
	logging.basicConfig(
		level=logging.INFO,
		filename="debug.log",
		filemode="w",
		format="%(asctime)s - %(levelname)s - %(message)s",
	)
	# Use an asyncio queue for non-blocking communication with the UI.
	inference_queue: asyncio.Queue[list[DetectionResult] | str] = asyncio.Queue()
	inference_config = InferenceConfig()  # type: ignore
	stop_event = threading.Event()

	inference_engine = InferenceEngine(inference_config, inference_queue)
	app = SiliconScopeApp(inference_queue, inference_engine)

	# Start the dummy feed thread. It will wait for the engine's started_event.
	feed_thread = threading.Thread(
		target=dummy_feed,
		args=(inference_engine, stop_event, inference_engine.started_event),
		daemon=True,
	)
	feed_thread.start()

	app.run()

	# Signal the dummy feed to stop and wait for it.
	stop_event.set()
	feed_thread.join()


if __name__ == "__main__":
	main()
