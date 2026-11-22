"""
Advanced Lightweight Web Server for FedShield Dashboard.
Uses Python's built-in ThreadingHTTPServer for asynchronous, concurrent request processing.
REST API Endpoints:
- GET  /                               : Interactive single-page dashboard
- GET  /api/summary                    : Summary records of all completed experiments
- GET  /api/experiment/<id>            : Full JSON trajectory & metrics of a specific experiment
- GET  /api/simulate_dirichlet         : Generates realistic Non-IID Dirichlet class distribution
- GET  /api/system_info                : System device, model parameters, and platform details
- GET  /api/export_csv                 : Downloads raw experiments_summary.csv
- POST /api/run                        : Triggers live federated training & returns real-time metrics
"""

import os
import sys
import json
import csv
import urllib.parse
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import numpy as np
import torch

# Add root directory to python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from config.config_parser import load_config, get_device
from models.cnn import create_model


class FedShieldHandler(BaseHTTPRequestHandler):

    def _set_headers(self, content_type: str = "application/json", status: int = 200, extra_headers: dict = None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, v)
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(status=204)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        # 1. Main Dashboard HTML
        if path in ["/", "/index.html"]:
            html_path = os.path.join(ROOT_DIR, "dashboard", "index.html")
            if os.path.exists(html_path):
                with open(html_path, "r", encoding="utf-8") as f:
                    content = f.read()
                self._set_headers(content_type="text/html; charset=utf-8", status=200)
                self.wfile.write(content.encode("utf-8"))
            else:
                self._set_headers(content_type="text/plain", status=404)
                self.wfile.write(b"index.html not found.")

        # 2. Summary CSV Data
        elif path == "/api/summary":
            csv_path = os.path.join(ROOT_DIR, "experiments", "results", "experiments_summary.csv")
            results = []
            if os.path.exists(csv_path):
                with open(csv_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        results.append(row)
            self._set_headers(content_type="application/json", status=200)
            self.wfile.write(json.dumps(results).encode("utf-8"))

        # 3. Experiment Detail JSON
        elif path.startswith("/api/experiment/"):
            exp_id = path.replace("/api/experiment/", "").strip()
            json_path = os.path.join(ROOT_DIR, "experiments", "results", f"{exp_id}.json")
            if os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._set_headers(content_type="application/json", status=200)
                self.wfile.write(json.dumps(data).encode("utf-8"))
            else:
                self._set_headers(content_type="application/json", status=404)
                self.wfile.write(json.dumps({"error": f"Experiment '{exp_id}' not found"}).encode("utf-8"))

        # 4. Dirichlet Distribution Simulator (NumPy Dirichlet sampling)
        elif path == "/api/simulate_dirichlet":
            alpha = float(query.get("alpha", [0.5])[0])
            num_clients = int(query.get("clients", [5])[0])
            seed = int(query.get("seed", [42])[0])

            np.random.seed(seed)
            num_classes = 10
            class_names = [
                "Airplane", "Automobile", "Bird", "Cat", "Deer",
                "Dog", "Frog", "Horse", "Ship", "Truck"
            ]
            samples_per_class = 500  # subsample count for rapid rendering
            matrix = np.zeros((num_clients, num_classes), dtype=int)

            for c in range(num_classes):
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                matrix[:, c] = (proportions * samples_per_class).astype(int)

            response = {
                "alpha": alpha,
                "clients": [f"Client {i}" for i in range(num_clients)],
                "classes": class_names,
                "distribution": matrix.tolist()
            }
            self._set_headers(content_type="application/json", status=200)
            self.wfile.write(json.dumps(response).encode("utf-8"))

        # 5. System Status & Model Specs
        elif path == "/api/system_info":
            try:
                cfg = load_config(os.path.join(ROOT_DIR, "config", "config.yaml"))
                device = get_device(cfg.system.device)
                model = create_model()
                param_count = model.count_parameters()
                size_bytes = model.get_model_size_bytes()

                info = {
                    "framework": "FedShield v1.0",
                    "device": str(device),
                    "cuda_available": torch.cuda.is_available(),
                    "pytorch_version": torch.__version__,
                    "model_name": cfg.model.name,
                    "param_count": param_count,
                    "model_size_kb": round(size_bytes / 1024, 2),
                    "model_size_mb": round(size_bytes / (1024 * 1024), 3),
                    "dataset": "CIFAR-10",
                    "num_classes": 10,
                    "status": "HEALTHY"
                }
            except Exception as e:
                info = {"status": "DEGRADED", "error": str(e)}

            self._set_headers(content_type="application/json", status=200)
            self.wfile.write(json.dumps(info).encode("utf-8"))

        # 6. Export CSV
        elif path == "/api/export_csv":
            csv_path = os.path.join(ROOT_DIR, "experiments", "results", "experiments_summary.csv")
            if os.path.exists(csv_path):
                with open(csv_path, "r", encoding="utf-8") as f:
                    csv_content = f.read()
                extra = {"Content-Disposition": "attachment; filename=experiments_summary.csv"}
                self._set_headers(content_type="text/csv", status=200, extra_headers=extra)
                self.wfile.write(csv_content.encode("utf-8"))
            else:
                self._set_headers(content_type="text/plain", status=404)
                self.wfile.write(b"No experiments_summary.csv found.")

        else:
            self._set_headers(content_type="text/plain", status=404)
            self.wfile.write(b"Not Found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/run":
            content_length = int(self.headers.get("Content-Length", 0))
            post_body = self.rfile.read(content_length)
            params = json.loads(post_body.decode("utf-8")) if post_body else {}

            alg = params.get("algorithm", "fedavg")
            partition = params.get("partition_type", "iid")
            alpha = float(params.get("alpha", 0.5))
            clients = int(params.get("clients", 5))
            rounds = int(params.get("rounds", 2))
            malicious = float(params.get("malicious_fraction", 0.0))
            attack = params.get("attack_type", "sign_flipping")
            aggregation = params.get("aggregation", "fedavg")
            dp = bool(params.get("dp_enabled", False))

            try:
                import time
                from experiments.federated import run_federated_experiment

                exp_id = f"web_{alg}_{partition}_{int(time.time())}"
                result = run_federated_experiment(
                    mode="quick",
                    algorithm=alg,
                    partition_type=partition,
                    dirichlet_alpha=alpha,
                    rounds=rounds,
                    num_clients=clients,
                    malicious_fraction=malicious,
                    attack_type=attack,
                    aggregation=aggregation,
                    dp_enabled=dp,
                    noise_multiplier=0.5 if dp else 0.0,
                    experiment_name=exp_id
                )
                final_metrics = result.get("final_metrics", {})
                rounds_history = result.get("rounds_history", [])

                self._set_headers(content_type="application/json", status=200)
                self.wfile.write(json.dumps({
                    "status": "COMPLETED",
                    "experiment_id": exp_id,
                    "final_accuracy": final_metrics.get("accuracy", 0.0),
                    "final_loss": final_metrics.get("loss", 0.0),
                    "final_f1": final_metrics.get("f1_score", 0.0),
                    "rounds": rounds_history
                }).encode("utf-8"))
            except Exception as e:
                self._set_headers(content_type="application/json", status=500)
                self.wfile.write(json.dumps({"status": "FAILED", "error": str(e)}).encode("utf-8"))
        else:
            self._set_headers(content_type="text/plain", status=404)
            self.wfile.write(b"Not Found")


def run_server(port: int = 8080):
    server_address = ("127.0.0.1", port)
    httpd = ThreadingHTTPServer(server_address, FedShieldHandler)
    print("=" * 65)
    print(f" 🛡️ FedShield Multi-Threaded Web Server Running at:")
    print(f" 👉 http://localhost:{port}")
    print("=" * 65)
    httpd.serve_forever()


if __name__ == "__main__":
    port = 8080
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    run_server(port)
