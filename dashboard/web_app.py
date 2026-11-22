"""
Lightweight Web Server for FedShield Frontend.
Built using standard library http.server (no external web framework dependencies needed).
Provides:
1. Static HTML/JS Dashboard Serving (http://localhost:8080)
2. REST API /api/summary for experiment benchmarking logs
3. REST API /api/run to execute live federated experiments directly from browser
"""

import os
import sys
import json
import csv
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

# Add root directory to python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)


class FedShieldHandler(BaseHTTPRequestHandler):

    def _set_headers(self, content_type: str = "application/json", status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(status=204)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ["/", "/index.html"]:
            html_path = os.path.join(ROOT_DIR, "dashboard", "index.html")
            if os.path.exists(html_path):
                with open(html_path, "r", encoding="utf-8") as f:
                    content = f.read()
                self._set_headers(content_type="text/html", status=200)
                self.wfile.write(content.encode("utf-8"))
            else:
                self._set_headers(content_type="text/plain", status=404)
                self.wfile.write(b"index.html not found.")

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
                self.wfile.write(json.dumps({"error": "Experiment not found"}).encode("utf-8"))

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
            malicious = float(params.get("malicious_fraction", 0.0))
            attack = params.get("attack_type", "sign_flipping")
            aggregation = params.get("aggregation", "fedavg")
            dp = bool(params.get("dp_enabled", False))

            try:
                from experiments.federated import run_federated_experiment
                import time
                result = run_federated_experiment(
                    mode="quick",
                    algorithm=alg,
                    partition_type=partition,
                    dirichlet_alpha=alpha,
                    rounds=2,
                    num_clients=clients,
                    malicious_fraction=malicious,
                    attack_type=attack,
                    aggregation=aggregation,
                    dp_enabled=dp,
                    noise_multiplier=0.5 if dp else 0.0,
                    experiment_name=f"web_{alg}_{int(time.time())}"
                )
                final_metrics = result.get("final_metrics", {})
                self._set_headers(content_type="application/json", status=200)
                self.wfile.write(json.dumps({
                    "status": "COMPLETED",
                    "final_accuracy": final_metrics.get("accuracy", 0.0),
                    "final_loss": final_metrics.get("loss", 0.0)
                }).encode("utf-8"))
            except Exception as e:
                self._set_headers(content_type="application/json", status=500)
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self._set_headers(content_type="text/plain", status=404)
            self.wfile.write(b"Not Found")


def run_server(port: int = 8080):
    server_address = ("127.0.0.1", port)
    httpd = HTTPServer(server_address, FedShieldHandler)
    print(f"============================================================")
    print(f" 🛡️ FedShield Web Frontend Server running on http://localhost:{port}")
    print(f"============================================================")
    httpd.serve_forever()


if __name__ == "__main__":
    port = 8080
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    run_server(port)
