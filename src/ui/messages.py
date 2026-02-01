"""Custom messages for the Textual UI."""

from textual.message import Message

from src.core.inference import DetectionResult


class InferenceResult(Message):
	"""Posted when a new inference result is available."""

	def __init__(self, results: list[DetectionResult]) -> None:
		self.results = results
		super().__init__()


class InferenceError(Message):
	"""Posted when an error occurs during inference."""

	def __init__(self, error_message: str) -> None:
		self.error_message = error_message
		super().__init__()
