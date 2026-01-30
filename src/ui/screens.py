"""Screens for the SiliconScope AOI application."""

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Footer, Header, Static


class MainScreen(Screen[None]):
	"""The main screen for the SiliconScope AOI application."""

	def compose(self) -> ComposeResult:
		yield Header()
		yield Footer()
		yield Static("Main Screen Content", id="main-screen-content")
