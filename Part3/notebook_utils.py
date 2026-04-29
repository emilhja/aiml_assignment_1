"""Shared utilities for Part 3 generated notebooks and report scripts."""

from pathlib import Path

import nbformat
from nbclient import NotebookClient


def markdown_cell(source):
    """Build a markdown notebook cell."""
    return nbformat.v4.new_markdown_cell(source=source)


def code_cell(source):
    """Build a code notebook cell."""
    return nbformat.v4.new_code_cell(source=source)


def write_notebook(notebook_path, cells):
    """Write a notebook with standard Python metadata and return its path."""
    notebook_path = Path(notebook_path)
    notebook = nbformat.v4.new_notebook()
    notebook.cells = cells
    notebook.metadata["language_info"] = {"name": "python"}
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook_path.write_text(
        nbformat.writes(notebook, version=4),
        encoding="utf-8",
    )
    return notebook_path


def execute_notebook(notebook_path, timeout=1200):
    """Execute a notebook in place so outputs are saved."""
    notebook_path = Path(notebook_path)
    notebook = nbformat.read(notebook_path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(notebook_path.parent)}},
    )
    client.execute()
    notebook_path.write_text(nbformat.writes(notebook, version=4), encoding="utf-8")
    return notebook_path
