"""
Generic Thread Management Utilities.

This module provides a context manager for safely starting and joining
any threading.Thread instance, ensuring proper resource cleanup.
"""

import threading
from types import TracebackType


class ThreadManager:
	"""
	A context manager to ensure a threading.Thread is started and then joined.

	Usage:
		my_thread = threading.Thread(target=my_function)
		with ThreadManager(my_thread):
			# my_thread is running here
			pass
		# my_thread is guaranteed to be joined here
	"""

	def __init__(self, thread: threading.Thread) -> None:
		"""
		Initializes the ThreadManager with the thread to manage.

		Args:
			thread: The threading.Thread instance to manage.
		"""
		self._thread = thread

	def __enter__(self) -> None:
		"""
		Starts the managed thread when entering the 'with' block.
		"""
		if not self._thread.is_alive():
			self._thread.start()

	def __exit__(
		self,
		exc_type: type[BaseException] | None,
		exc_val: BaseException | None,
		exc_tb: TracebackType | None,
	) -> None:
		"""
		Joins the managed thread when exiting the 'with' block,
		guaranteeing cleanup.
		"""
		if self._thread.is_alive():
			self._thread.join()
