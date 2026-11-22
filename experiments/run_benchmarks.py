"""
Benchmark Suite Runner for FedShield.
Executes four comparative FL experiments for research, benchmarking, and demonstration:
1. Benchmark A: Data Heterogeneity (FedAvg IID vs FedAvg Non-IID Dirichlet alpha=0.5).
2. Benchmark B: Algorithmic Robustness under Non-IID (FedAvg vs FedProx alpha=0.1).
3. Benchmark C: Byzantine Attacks & Defenses (Sign Flipping with FedAvg vs Trimmed Mean vs Coordinate Median).
4. Benchmark D: Differential Privacy (Privacy-Utility Tradeoff at noise=0.0 vs 0.5 vs 1.0).
"""

import os
import sys
import argparse
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.federated import run_federated_experiment


def run_all_benchmarks(mode: str = "quick"):
    """Runs a complete set of comparative research experiments."""
    print("=" * 70)
    print(f" Starting FedShield Benchmark Suite [Mode: {mode.upper()}]")
    print("=" * 70)
    start_time = time.time()

    rounds = 3 if mode == "quick" else 5

    # 1. Benchmark A: IID vs Non-IID Dirichlet
    print("\n[Benchmark 1/4] Running FedAvg on IID data...")
    run_federated_experiment(
        mode=mode,
        algorithm="fedavg",
        partition_type="iid",
        rounds=rounds,
        num_clients=5 if mode == "quick" else 10,
        experiment_name=f"benchmark_fedavg_iid_{mode}"
    )

    print("\n[Benchmark 1/4 (cont.)] Running FedAvg on Non-IID data (alpha=0.5)...")
    run_federated_experiment(
        mode=mode,
        algorithm="fedavg",
        partition_type="non_iid_dirichlet",
        dirichlet_alpha=0.5,
        rounds=rounds,
        num_clients=5 if mode == "quick" else 10,
        experiment_name=f"benchmark_fedavg_non_iid_0.5_{mode}"
    )

    # 2. Benchmark B: FedProx on Extreme Non-IID
    print("\n[Benchmark 2/4] Running FedProx on Extreme Non-IID (alpha=0.1)...")
    run_federated_experiment(
        mode=mode,
        algorithm="fedprox",
        partition_type="non_iid_dirichlet",
        dirichlet_alpha=0.1,
        rounds=rounds,
        num_clients=5 if mode == "quick" else 10,
        experiment_name=f"benchmark_fedprox_non_iid_0.1_{mode}"
    )

    # 3. Benchmark C: Byzantine Defense Comparison (Attack: Sign Flipping, 20% malicious)
    print("\n[Benchmark 3/4] Running Byzantine Attack on Standard FedAvg (Sign Flipping, 20% Malicious)...")
    run_federated_experiment(
        mode=mode,
        algorithm="fedavg",
        partition_type="iid",
        rounds=rounds,
        num_clients=5 if mode == "quick" else 10,
        malicious_fraction=0.2,
        attack_type="sign_flipping",
        aggregation="fedavg",
        experiment_name=f"benchmark_attack_fedavg_vulnerable_{mode}"
    )

    print("\n[Benchmark 3/4 (cont.)] Defending Byzantine Attack with Coordinate Median...")
    run_federated_experiment(
        mode=mode,
        algorithm="fedavg",
        partition_type="iid",
        rounds=rounds,
        num_clients=5 if mode == "quick" else 10,
        malicious_fraction=0.2,
        attack_type="sign_flipping",
        aggregation="coordinate_median",
        experiment_name=f"benchmark_defense_coordinate_median_{mode}"
    )

    print("\n[Benchmark 3/4 (cont.)] Defending Byzantine Attack with Trimmed Mean...")
    run_federated_experiment(
        mode=mode,
        algorithm="fedavg",
        partition_type="iid",
        rounds=rounds,
        num_clients=5 if mode == "quick" else 10,
        malicious_fraction=0.2,
        attack_type="sign_flipping",
        aggregation="trimmed_mean",
        experiment_name=f"benchmark_defense_trimmed_mean_{mode}"
    )

    # 4. Benchmark D: Differential Privacy
    print("\n[Benchmark 4/4] Running Privacy-Preserving FL with DP-FedAvg (noise=0.5)...")
    run_federated_experiment(
        mode=mode,
        algorithm="fedavg",
        partition_type="iid",
        rounds=rounds,
        num_clients=5 if mode == "quick" else 10,
        dp_enabled=True,
        noise_multiplier=0.5,
        experiment_name=f"benchmark_dp_fedavg_noise_0.5_{mode}"
    )

    elapsed = round(time.time() - start_time, 2)
    print("\n" + "=" * 70)
    print(f" FedShield Benchmark Suite Finished Successfully in {elapsed}s!")
    print(f" All results recorded in: experiments/results/experiments_summary.csv")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FedShield Comparative Benchmark Suite")
    parser.add_argument("--mode", type=str, default="quick", choices=["quick", "full"], help="quick or full benchmark")
    args = parser.parse_args()
    run_all_benchmarks(mode=args.mode)
