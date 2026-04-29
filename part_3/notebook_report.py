"""Compatibility wrappers for Part 3 Jupyter notebook reports."""

try:
    from .notebook_templates import create_experiment_report_notebook
    from .notebook_utils import execute_notebook
except ImportError:  # pragma: no cover - supports direct script execution
    from notebook_templates import create_experiment_report_notebook
    from notebook_utils import execute_notebook


def create_report_notebook(run_dir):
    """Create a runnable notebook report for a Part 3 experiment run."""
    return create_experiment_report_notebook(run_dir)


def execute_report_notebook(notebook_path, timeout=1200):
    """Execute a notebook in place so outputs are saved."""
    return execute_notebook(notebook_path, timeout=timeout)
