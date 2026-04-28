"""Shared utilities for Part 2 generated notebooks and report scripts."""

import json
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


def execute_notebook(notebook_path, timeout=600):
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


def load_json(path):
    """Load a UTF-8 JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def summarize_run_paths(summaries):
    """Return run path metadata from ``(run_name, summary_path, summary)`` tuples."""
    run_paths = {
        run_name: str(summary_path.parent)
        for run_name, summary_path, _summary in summaries
    }
    ordered_run_names = [
        run_name for run_name, _summary_path, _summary in summaries
    ]
    return run_paths, ordered_run_names
