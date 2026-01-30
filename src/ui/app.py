from textual.app import App, ComposeResult
from textual.reactive import var
from textual.widgets import Footer, Header, Static


class SiliconScopeApp(App[None]):
	"""The main application for SiliconScope AOI."""

	BINDINGS = [
		("^q", "quit", "Quit"),
	]

	dark = var(True)

	def compose(self) -> ComposeResult:
		"""Create child widgets for the app."""
		yield Header()
		yield Footer()
		yield Static("Hello from SiliconScope!", id="main-content")

	async def action_quit(self) -> None:
		"""An action to quit the app."""
		self.exit()


if __name__ == "__main__":
	app = SiliconScopeApp()
	app.run()
