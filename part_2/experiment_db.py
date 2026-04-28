#!/usr/bin/env python3
"""SQLite utilities for storing experiment runs and per-epoch metrics."""

import sqlite3
from pathlib import Path


class ExperimentDB:
    """Persist experiment metadata and metrics for later comparison."""

    METRIC_DECIMALS = 4
    DURATION_DECIMALS = 2
    CONFIG_DECIMALS = 6

    def __init__(self, db_path):
        """Open the database connection and ensure required tables exist."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.db_path)
        self.connection.execute("PRAGMA foreign_keys = ON")
        self._create_tables()

    @classmethod
    def _round_metric(cls, value):
        """Round losses and accuracies before persisting them to SQLite."""
        return round(float(value), cls.METRIC_DECIMALS)

    @classmethod
    def _round_duration(cls, value):
        """Round elapsed-time values before persisting them to SQLite."""
        return round(float(value), cls.DURATION_DECIMALS)

    @classmethod
    def _round_config_decimal(cls, value):
        """Round small configuration floats while preserving useful precision."""
        return round(float(value), cls.CONFIG_DECIMALS)

    def _create_tables(self):
        """Create tables used to compare runs and epoch-level metrics."""
        cursor = self.connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_name TEXT NOT NULL,
                started_at TEXT NOT NULL,
                model_name TEXT NOT NULL,
                batch_size INTEGER NOT NULL,
                epochs_requested INTEGER NOT NULL,
                epochs_completed INTEGER,
                learning_rate REAL NOT NULL,
                checkpoint_interval INTEGER NOT NULL,
                validation_ratio REAL NOT NULL,
                seed INTEGER NOT NULL,
                early_stopping_patience INTEGER NOT NULL,
                optimizer_name TEXT NOT NULL,
                weight_decay REAL NOT NULL,
                hidden_size_1 INTEGER,
                hidden_size_2 INTEGER,
                augmentation_enabled INTEGER NOT NULL,
                augmentation_description TEXT NOT NULL,
                device TEXT NOT NULL,
                output_dir TEXT NOT NULL,
                git_commit TEXT,
                git_is_dirty INTEGER,
                best_epoch INTEGER,
                best_validation_loss REAL,
                best_validation_accuracy REAL,
                final_test_loss REAL,
                final_test_accuracy REAL,
                time_to_best_model_seconds REAL,
                total_training_time_seconds REAL,
                average_epoch_time_seconds REAL,
                stopped_early INTEGER,
                best_model_path TEXT
            )
            """
        )
        self._ensure_column("runs", "git_commit", "TEXT")
        self._ensure_column("runs", "git_is_dirty", "INTEGER")
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS epoch_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                epoch INTEGER NOT NULL,
                train_loss REAL NOT NULL,
                train_accuracy REAL NOT NULL,
                val_loss REAL NOT NULL,
                val_accuracy REAL NOT NULL,
                epoch_duration_seconds REAL NOT NULL,
                is_best_epoch INTEGER NOT NULL,
                checkpoint_saved INTEGER NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            )
            """
        )
        self.connection.commit()

    def _ensure_column(self, table_name, column_name, column_type):
        """Add a column to an existing table when upgrading an older database."""
        cursor = self.connection.cursor()
        existing_columns = {
            row[1]
            for row in cursor.execute(f"PRAGMA table_info({table_name})").fetchall()
        }
        if column_name in existing_columns:
            return
        cursor.execute(
            f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"
        )
        self.connection.commit()

    def create_run(self, run_name, started_at, config):
        """Insert an experiment run row and return its database identifier."""
        cursor = self.connection.cursor()
        cursor.execute(
            """
            INSERT INTO runs (
                run_name,
                started_at,
                model_name,
                batch_size,
                epochs_requested,
                learning_rate,
                checkpoint_interval,
                validation_ratio,
                seed,
                early_stopping_patience,
                optimizer_name,
                weight_decay,
                hidden_size_1,
                hidden_size_2,
                augmentation_enabled,
                augmentation_description,
                device,
                output_dir,
                git_commit,
                git_is_dirty
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_name,
                started_at,
                config["model_name"],
                config["batch_size"],
                config["epochs"],
                self._round_config_decimal(config["learning_rate"]),
                config["checkpoint_interval"],
                self._round_config_decimal(config["validation_ratio"]),
                config["seed"],
                config["early_stopping_patience"],
                config["optimizer_name"],
                self._round_config_decimal(config["weight_decay"]),
                config.get("hidden_size_1"),
                config.get("hidden_size_2"),
                int(config["augmentation_enabled"]),
                config["augmentation_description"],
                config["device"],
                config["output_dir"],
                config.get("git_commit"),
                (
                    int(config["git_is_dirty"])
                    if config.get("git_is_dirty") is not None
                    else None
                ),
            ),
        )
        self.connection.commit()
        return cursor.lastrowid

    def log_epoch(
        self,
        run_id,
        epoch,
        train_loss,
        train_accuracy,
        val_loss,
        val_accuracy,
        epoch_duration_seconds,
        is_best_epoch,
        checkpoint_saved,
    ):
        """Insert per-epoch metrics for a run."""
        self.connection.execute(
            """
            INSERT INTO epoch_metrics (
                run_id,
                epoch,
                train_loss,
                train_accuracy,
                val_loss,
                val_accuracy,
                epoch_duration_seconds,
                is_best_epoch,
                checkpoint_saved
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                epoch,
                self._round_metric(train_loss),
                self._round_metric(train_accuracy),
                self._round_metric(val_loss),
                self._round_metric(val_accuracy),
                self._round_duration(epoch_duration_seconds),
                int(is_best_epoch),
                int(checkpoint_saved),
            ),
        )
        self.connection.commit()

    def finalize_run(self, run_id, summary):
        """Update a run row with final results after training completes."""
        self.connection.execute(
            """
            UPDATE runs
            SET epochs_completed = ?,
                best_epoch = ?,
                best_validation_loss = ?,
                best_validation_accuracy = ?,
                final_test_loss = ?,
                final_test_accuracy = ?,
                time_to_best_model_seconds = ?,
                total_training_time_seconds = ?,
                average_epoch_time_seconds = ?,
                stopped_early = ?,
                best_model_path = ?
            WHERE run_id = ?
            """,
            (
                summary["epochs_completed"],
                summary["best_epoch"],
                self._round_metric(summary["best_validation_loss"]),
                self._round_metric(summary["best_validation_accuracy"]),
                self._round_metric(summary["final_test_loss"]),
                self._round_metric(summary["final_test_accuracy"]),
                self._round_duration(summary["time_to_best_model_seconds"]),
                self._round_duration(summary["total_training_time_seconds"]),
                self._round_duration(summary["average_epoch_time_seconds"]),
                int(summary["stopped_early"]),
                summary["best_model_path"],
                run_id,
            ),
        )
        self.connection.commit()

    def close(self):
        """Close the SQLite connection."""
        self.connection.close()
