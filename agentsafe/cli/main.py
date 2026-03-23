from __future__ import annotations

import runpy
import sys
from pathlib import Path

import typer

app = typer.Typer(help="AgentSafe — runtime safety monitoring for AI agents")


@app.command()
def run(
    script: Path = typer.Argument(..., help="Python agent script to run"),
    db_path: str = typer.Option(".agentsafe/events.db", help="Path to SQLite database"),
):
    """Run a Python agent script with AgentSafe monitoring injected."""
    import agentsafe

    if not script.exists():
        typer.echo(f"Error: {script} not found", err=True)
        raise typer.Exit(1)

    # Inject agentsafe into the script's globals so it can call agentsafe.wrap()
    # without any import changes, and override the default db_path
    original_wrap = agentsafe.wrap

    def patched_wrap(client, **kwargs):
        kwargs.setdefault("db_path", db_path)
        return original_wrap(client, **kwargs)

    sys.argv = [str(script)] + sys.argv[2:]
    runpy.run_path(str(script), init_globals={"agentsafe": agentsafe}, run_name="__main__")


@app.command()
def dashboard(
    db_path: str = typer.Option(".agentsafe/events.db", help="Path to SQLite database"),
    host: str = typer.Option("127.0.0.1", help="Host to bind to"),
    port: int = typer.Option(7777, help="Port to listen on"),
):
    """Launch the AgentSafe dashboard web UI."""
    import uvicorn

    from agentsafe.dashboard.app import create_app
    from agentsafe.storage.database import Database
    from agentsafe.storage.repository import EventRepository

    db = Database(db_path)
    repo = EventRepository(db)
    fastapi_app = create_app(repo)

    typer.echo(f"AgentSafe dashboard → http://{host}:{port}")
    uvicorn.run(fastapi_app, host=host, port=port, log_level="warning")


@app.command()
def stats(
    db_path: str = typer.Option(".agentsafe/events.db", help="Path to SQLite database"),
):
    """Print summary stats from the event store."""
    from agentsafe.storage.database import Database
    from agentsafe.storage.repository import EventRepository

    db = Database(db_path)
    repo = EventRepository(db)
    s = repo.stats()
    typer.echo(f"Total events : {s['total_events']}")
    typer.echo(f"Flagged      : {s['flagged_events']}")
    for sev, count in s.get("by_severity", {}).items():
        typer.echo(f"  {sev:<10}: {count}")
