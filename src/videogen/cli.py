from __future__ import annotations

import logging
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from videogen.config import get_settings
from videogen.pipeline import Pipeline

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


@app.command()
def health() -> None:
    """Check Ollama and ComfyUI."""
    pipe = Pipeline()
    h = pipe.health_check()
    table = Table(title="Local services")
    table.add_column("Service")
    table.add_column("OK")
    for k, v in h.items():
        table.add_row(k, "yes" if v else "NO")
    console.print(table)
    if not all(h.values()):
        raise typer.Exit(code=1)


@app.command()
def run(
    topic: Optional[str] = typer.Option(None, help="Topic text"),
    pick_topic: bool = typer.Option(False, help="Let local model pick a topic"),
    duration: int = typer.Option(60, help="Target duration seconds"),
    style: str = typer.Option(
        "sketch",
        help="Style preset: sketch|classic_cartoon|storybook|retro_digital|painterly|graphic",
    ),
    run_id: Optional[str] = typer.Option(None, help="Resume this run id"),
    no_resume: bool = typer.Option(False, help="Ignore prior stage status"),
    skip_qa: bool = typer.Option(False, help="Skip vision QA"),
    quality_writer: bool = typer.Option(False, help="Use larger local writer model"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the full local video pipeline."""
    _setup_logging(verbose)
    settings = get_settings()
    if quality_writer:
        settings.use_quality_writer = True
    if not topic and not pick_topic:
        console.print("[red]Provide --topic or --pick-topic[/red]")
        raise typer.Exit(code=2)

    pipe = Pipeline(settings)

    def on_progress(stage: str, state: str) -> None:
        console.print(f"[bold]{stage}[/bold]: {state}")

    store = pipe.run(
        topic=topic,
        duration_s=duration,
        style_preset=style,
        pick_topic=pick_topic,
        run_id=run_id,
        resume=not no_resume,
        skip_vision_qa=skip_qa,
        on_progress=on_progress,
    )
    final = store.path("final.mp4")
    console.print(f"[green]Done[/green] run={store.run_id}")
    if final.exists():
        console.print(f"Video: {final}")


@app.command()
def serve(
    host: str = "0.0.0.0",
    port: int = 7860,
) -> None:
    """Start the web UI (default: all interfaces, LAN-reachable)."""
    import uvicorn

    console.print(f"Serving on http://{host}:{port} (reachable on your LAN)")
    uvicorn.run("videogen.webapp:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    app()
