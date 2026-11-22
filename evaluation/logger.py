"""
Experiment Logger for FedShield.
Persists complete experiment parameters, round trajectories, communication overhead,
and final metrics to formatted JSON and summary CSV files.
"""

import os
import json
import csv
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import asdict, is_dataclass


class ExperimentLogger:
    """Manages experiment metadata, round metrics recording, and disk persistence."""

    def __init__(self, results_dir: str = "./experiments/results", experiment_name: Optional[str] = None):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = experiment_name or f"exp_{timestamp_str}"
        self.json_path = os.path.join(self.results_dir, f"{self.experiment_id}.json")
        self.summary_csv_path = os.path.join(self.results_dir, "experiments_summary.csv")

        self.data: Dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "timestamp": datetime.now().isoformat(),
            "config": {},
            "rounds": [],
            "final_metrics": {},
            "training_time_seconds": 0.0,
            "status": "RUNNING"
        }
        self.start_time = time.time()

    def set_config(self, config_obj: Any) -> None:
        """Stores serializable configuration dictionary."""
        if is_dataclass(config_obj):
            self.data["config"] = asdict(config_obj)
        elif isinstance(config_obj, dict):
            self.data["config"] = config_obj
        else:
            self.data["config"] = str(config_obj)

    def log_round(self, round_data: Dict[str, Any]) -> None:
        """Logs a single communication or training round."""
        self.data["rounds"].append(round_data)

    def finalize(self, final_metrics: Dict[str, Any], status: str = "COMPLETED") -> str:
        """Calculates elapsed time, sets final metrics, and writes JSON and CSV."""
        self.data["training_time_seconds"] = round(time.time() - self.start_time, 2)
        self.data["final_metrics"] = final_metrics
        self.data["status"] = status

        # Save individual JSON report
        with open(self.json_path, "w") as f:
            json.dump(self.data, f, indent=2)

        # Append to summary CSV
        self._update_summary_csv()

        print(f"[FedShield Logger] Results successfully saved to: {self.json_path}")
        return self.json_path

    def _update_summary_csv(self) -> None:
        """Appends one-line summary of this experiment to CSV for cross-run comparison."""
        file_exists = os.path.exists(self.summary_csv_path)
        cfg = self.data.get("config", {})

        # Extract nested attributes safely
        fed_cfg = cfg.get("federated", {})
        sec_cfg = cfg.get("security", {})
        dp_cfg = cfg.get("privacy", {})
        part_cfg = cfg.get("partition", {})
        final = self.data.get("final_metrics", {})

        row = {
            "experiment_id": self.experiment_id,
            "timestamp": self.data.get("timestamp", ""),
            "algorithm": fed_cfg.get("algorithm", cfg.get("algorithm", "centralized")),
            "num_clients": fed_cfg.get("num_clients", 1),
            "partition_type": part_cfg.get("type", "N/A"),
            "dirichlet_alpha": part_cfg.get("dirichlet_alpha", "N/A"),
            "malicious_fraction": sec_cfg.get("malicious_client_fraction", 0.0),
            "aggregation_method": sec_cfg.get("aggregation_method", "fedavg"),
            "dp_enabled": dp_cfg.get("dp_enabled", False),
            "noise_multiplier": dp_cfg.get("noise_multiplier", 0.0),
            "final_accuracy": final.get("accuracy", "Not yet measured"),
            "final_loss": final.get("loss", "Not yet measured"),
            "final_f1": final.get("f1_score", "Not yet measured"),
            "training_time_s": self.data.get("training_time_seconds", 0.0),
            "status": self.data.get("status", "UNKNOWN")
        }

        fieldnames = list(row.keys())
        with open(self.summary_csv_path, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
