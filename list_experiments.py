#!/usr/bin/env python3
"""Print a compact summary of recorded experiment runs."""

import sqlite3
from pathlib import Path


def list_experiments(db_path):
    """Read the experiment database and print the most useful fields."""
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    cursor.execute(
        """
        SELECT
            run_id,
            run_name,
            model_name,
            learning_rate,
            batch_size,
            best_epoch,
            best_validation_loss,
            final_test_accuracy,
            total_training_time_seconds
        FROM runs
        ORDER BY run_id DESC
        """
    )
    rows = cursor.fetchall()
    connection.close()

    if not rows:
        print("No experiment runs found.")
        return

    for row in rows:
        (
            run_id,
            run_name,
            model_name,
            learning_rate,
            batch_size,
            best_epoch,
            best_validation_loss,
            final_test_accuracy,
            total_training_time_seconds,
        ) = row
        print(
            f"run_id={run_id} | run={run_name} | model={model_name} | "
            f"lr={learning_rate} | batch_size={batch_size} | "
            f"best_epoch={best_epoch} | "
            f"best_val_loss={best_validation_loss:.4f} | "
            f"test_acc={final_test_accuracy:.2%} | "
            f"total_time={total_training_time_seconds:.2f}s"
        )


if __name__ == "__main__":
    default_db = Path("outputs/Part2/experiments.db")
    list_experiments(default_db)
