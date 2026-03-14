"""CLI entry point: spatialxgene launch <file.h5ad>"""

import click
import signal
import socket
import sys
import time


def _find_free_port(host: str, start: int, max_tries: int = 20) -> int:
    """Return the first port >= start that can actually be bound."""
    for port in range(start, start + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    raise RuntimeError(
        f'No free port found in range {start}–{start + max_tries - 1}. '
        'Pass --port to choose a different base port.'
    )


@click.group()
def main():
    """spatialxgene — interactive spatial transcriptomics viewer."""


@main.command()
@click.argument('h5ad_file', type=click.Path(exists=True, dir_okay=False))
@click.option('--host',      default='127.0.0.1', show_default=True, help='Host to bind.')
@click.option('--port',      default=8050,        show_default=True, help='Preferred port (auto-increments if busy).')
@click.option('--subsample', default=None, type=int, metavar='N',
              help='Randomly subsample to N cells (speeds up large datasets).')
@click.option('--seed',      default=42,          show_default=True, help='Random seed for subsampling.')
@click.option('--debug',        is_flag=True,    help='Run Dash in debug mode.')
@click.option('--skip-columns', default=None, metavar='COLS',
              help='Comma-separated column names to hide from the Color By dropdown.')
@click.option('--no-shift-libraries', is_flag=True,
              help='Disable auto-shifting of overlapping library spatial coordinates.')
def launch(h5ad_file, host, port, subsample, seed, debug, skip_columns, no_shift_libraries):
    """Launch the spatialxgene viewer for H5AD_FILE."""
    from .data import SpatialData
    from .app  import create_app

    # Find a port we can actually bind to (increments if preferred port is busy)
    actual_port = _find_free_port(host, port)
    if actual_port != port:
        click.echo(f'  Port {port} is in use — using {actual_port} instead.', err=True)

    skip_cols = set(c.strip() for c in skip_columns.split(',')) if skip_columns else None

    click.echo(f'Loading {h5ad_file} …', err=True)
    data = SpatialData(h5ad_file, subsample=subsample, seed=seed, skip_columns=skip_cols,
                       shift_libraries=not no_shift_libraries)
    click.echo(
        f'  {data.n_cells:,} cells'
        + (f' (subsampled from {data.n_cells_total:,})' if data.n_cells < data.n_cells_total else '')
        + f'  |  views: {[v["value"] for v in data.available_views()]}',
        err=True,
    )

    app = create_app(data)

    url = f'http://{host}:{actual_port}'
    click.echo(f'\n  spatialxgene running →  {url}\n  Press Ctrl-C to quit.\n', err=True)

    # Ensure clean exit on Ctrl-C
    def _shutdown(signum, frame):
        click.echo('\n  Shutting down…', err=True)
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Open browser automatically
    import threading, webbrowser
    def _open():
        time.sleep(1.2)
        webbrowser.open(url)
    threading.Thread(target=_open, daemon=True).start()

    app.run(host=host, port=actual_port, debug=debug)
