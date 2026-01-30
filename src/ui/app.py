from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.reactive import var
from textual.widgets import Footer, Header, Static


class SiliconScopeApp(App[None]):
	"""The main application for SiliconScope AOI."""

	CSS_PATH = "app.css"

	BINDINGS = [
		("q", "quit", "Quit"),
	]

	dark = var(True)

	def compose(self) -> ComposeResult:
		"""Create child widgets for the app."""
		yield Header()
		yield Footer()
		with Vertical(id="main-container"):
			yield Static("Video Feed Placeholder", id="video-feed")
			yield Static("Log View Placeholder", id="log-view")

	async def action_quit(self) -> None:
		"""An action to quit the app."""
		self.exit()


if __name__ == "__main__":
	app = SiliconScopeApp()
	app.run()
