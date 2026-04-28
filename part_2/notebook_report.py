#!/usr/bin/env python3
"""Compatibility wrappers for per-run Jupyter notebook reports."""

try:
    from .notebook_templates import create_experiment_report_notebook
    from .notebook_utils import execute_notebook
except ImportError:
    if __package__:
        raise
    from notebook_templates import create_experiment_report_notebook
    from notebook_utils import execute_notebook


def create_report_notebook(run_dir):
    """Create a runnable notebook report for an experiment run."""
    return create_experiment_report_notebook(run_dir)


def execute_report_notebook(notebook_path, timeout=600):
    """Execute a notebook in place so outputs are saved."""
    return execute_notebook(notebook_path, timeout=timeout)
