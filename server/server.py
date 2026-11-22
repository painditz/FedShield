"""
Federated Server Orchestrator for FedShield.
Manages the global model lifecycle, client sampling, round coordination,
Byzantine attack dispatching, robust aggregation, differential privacy,
global evaluation, and experiment persistence.
"""

from typing import List, Dict, Any, Optional
import time
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.config_parser import FedShieldConfig
from models.cnn import create_model
from clients.client import Client
from clients.partitioner import FederatedPartitioner
from attacks.poisoning import apply_attack
from server.aggregation import aggregate
from privacy.differential_privacy import DifferentialPrivacyManager
from evaluation.metrics import evaluate_model, compute_communication_cost
from evaluation.logger import ExperimentLogger
from algorithms.federated_algorithm import get_algorithm


class FederatedServer:
    """
    Central Coordinator for Federated Learning experiments in FedShield.
    """

    def __init__(
        self,
        config: FedShieldConfig,
        train_dataset: Any,
        test_dataset: Any,
        device: torch.device,
        experiment_id: Optional[str] = None
    ):
        self.config = config
        self.device = device
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        # 1. Global Model Initialization
        self.global_model = create_model(
            model_name=config.model.name,
            in_channels=config.model.in_channels,
            num_classes=config.model.num_classes
        ).to(self.device)

        self.model_size_bytes = self.global_model.get_model_size_bytes()
        self.criterion = nn.CrossEntropyLoss()

        # Global test loader
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=config.centralized.batch_size,
            shuffle=False,
            num_workers=config.system.num_workers
        )

        # 2. Partition Dataset among Clients
        self.partitioner = FederatedPartitioner(
            dataset=self.train_dataset,
            num_clients=config.federated.num_clients,
            partition_type=config.partition.type,
            dirichlet_alpha=config.partition.dirichlet_alpha,
            imbalance=config.partition.imbalance,
            seed=config.system.random_seed
        )

        # 3. Determine Malicious Clients for Byzantine Attacks
        num_malicious = int(np.round(
            config.security.malicious_client_fraction * config.federated.num_clients
        ))
        np.random.seed(config.system.random_seed)
        malicious_indices = set(np.random.choice(
            config.federated.num_clients, size=num_malicious, replace=False
        )) if num_malicious > 0 else set()
        self.malicious_client_ids = malicious_indices

        # 4. Instantiate Clients
        self.clients: List[Client] = []
        for i in range(config.federated.num_clients):
            client_loader = self.partitioner.get_client_dataloader(
                client_id=i,
                batch_size=config.federated.local_batch_size,
                shuffle=True,
                num_workers=config.system.num_workers
            )
            is_mal = (i in malicious_indices)
            client = Client(
                client_id=i,
                dataloader=client_loader,
                model_template=self.global_model,
                device=self.device,
                is_malicious=is_mal
            )
            self.clients.append(client)

        # 5. Algorithm & Privacy Setup
        self.algorithm = get_algorithm(
            name=config.federated.algorithm,
            fedprox_mu=config.federated.fedprox_mu
        )
        self.dp_manager = DifferentialPrivacyManager(
            dp_enabled=config.privacy.dp_enabled,
            max_grad_norm=config.privacy.max_grad_norm,
            noise_multiplier=config.privacy.noise_multiplier
        )

        # 6. Logger Setup
        exp_name = experiment_id or (
            f"fed_{config.federated.algorithm}_{config.partition.type}_"
            f"sec_{config.security.aggregation_method}_{int(time.time())}"
        )
        self.logger = ExperimentLogger(
            results_dir=config.logging.results_dir,
            experiment_name=exp_name
        )
        self.logger.set_config(config)

    def select_clients(self) -> List[Client]:
        """Samples participating clients based on client_fraction."""
        num_to_sample = max(1, int(
            self.config.federated.client_fraction * len(self.clients)
        ))
        if num_to_sample == len(self.clients):
            return self.clients
        selected_indices = np.random.choice(len(self.clients), size=num_to_sample, replace=False)
        return [self.clients[idx] for idx in selected_indices]

    def run_round(self, round_num: int) -> Dict[str, Any]:
        """
        Executes a single communication round:
        1. Select clients.
        2. Broadcast global parameters to clients.
        3. Local training on each client.
        4. Byzantine attacks injected on malicious clients.
        5. Robust aggregation on the server.
        6. Differential privacy noise injection (if enabled).
        7. Global model update and evaluation.
        """
        global_params = self.global_model.get_parameters()
        selected_clients = self.select_clients()

        client_updates: List[Dict[str, torch.Tensor]] = []
        sample_counts: List[int] = []
        round_losses: List[float] = []
        round_accs: List[float] = []

        # Local training loop across clients
        for client in selected_clients:
            train_result = self.algorithm.train_client(
                client=client,
                global_parameters=global_params,
                local_epochs=self.config.federated.local_epochs,
                learning_rate=self.config.federated.local_learning_rate,
                weight_decay=self.config.federated.weight_decay,
                dp_config=self.config.privacy
            )

            client_params = train_result["parameters"]
            # Apply Byzantine attack if client is malicious
            if client.is_malicious:
                client_params = apply_attack(
                    local_parameters=client_params,
                    global_parameters=global_params,
                    attack_type=self.config.security.attack_type,
                    scaling_factor=self.config.security.scaling_factor
                )

            client_updates.append(client_params)
            sample_counts.append(train_result["sample_count"])
            round_losses.append(train_result["train_loss"])
            round_accs.append(train_result["train_accuracy"])

        # Aggregate updates
        num_malicious_in_round = sum(1 for c in selected_clients if c.is_malicious)
        aggregated_weights = aggregate(
            updates=client_updates,
            method=self.config.security.aggregation_method,
            sample_counts=sample_counts,
            trimmed_mean_beta=self.config.security.trimmed_mean_beta,
            num_malicious=max(1, num_malicious_in_round)
        )

        # Apply Server-Side Differential Privacy Noise if enabled
        if self.config.privacy.dp_enabled:
            aggregated_weights = self.dp_manager.apply_differential_privacy(
                aggregated_weights=aggregated_weights,
                num_participating_clients=len(selected_clients)
            )

        # Update global model
        self.global_model.set_parameters(aggregated_weights)

        # Global Evaluation on Test Set
        eval_metrics = evaluate_model(
            model=self.global_model,
            data_loader=self.test_loader,
            criterion=self.criterion,
            device=self.device
        )

        # Calculate communication cost
        comm_stats = compute_communication_cost(
            model_size_bytes=self.model_size_bytes,
            num_participating_clients=len(selected_clients),
            num_rounds=1
        )

        # Privacy accounting
        sample_rate = len(selected_clients) / float(self.config.federated.num_clients)
        privacy_budget = self.dp_manager.get_privacy_budget(sample_rate=sample_rate)

        round_record = {
            "round": round_num,
            "participating_clients": len(selected_clients),
            "malicious_clients": num_malicious_in_round,
            "client_mean_train_loss": round(float(np.mean(round_losses)), 4),
            "client_mean_train_acc": round(float(np.mean(round_accs)), 2),
            "test_loss": eval_metrics["loss"],
            "test_accuracy": eval_metrics["accuracy"],
            "test_f1": eval_metrics["f1_score"],
            "test_precision": eval_metrics["precision"],
            "test_recall": eval_metrics["recall"],
            "mb_round_comm": comm_stats["mb_per_round"],
            "cumulative_mb": round(comm_stats["mb_per_round"] * round_num, 2),
            "privacy": privacy_budget
        }

        self.logger.log_round(round_record)
        return round_record

    def train(self, num_rounds: Optional[int] = None) -> Dict[str, Any]:
        """Runs complete federated learning process over all communication rounds."""
        rounds = num_rounds or self.config.federated.communication_rounds
        print("=" * 65)
        print(f" FedShield Federated Training Orchestrator")
        print(f" Algorithm     : {self.config.federated.algorithm.upper()}")
        print(f" Aggregation   : {self.config.security.aggregation_method}")
        print(f" Partition     : {self.config.partition.type} (alpha={self.config.partition.dirichlet_alpha})")
        print(f" Clients       : {self.config.federated.num_clients} (Malicious: {len(self.malicious_client_ids)})")
        print(f" Privacy (DP)  : {'ENABLED' if self.config.privacy.dp_enabled else 'DISABLED'}")
        print(f" Total Rounds  : {rounds}")
        print("=" * 65)

        best_accuracy = 0.0
        best_metrics = {}

        pbar = tqdm(range(1, rounds + 1), desc="Federated Rounds")
        for r in pbar:
            round_stats = self.run_round(r)
            acc = round_stats["test_accuracy"]
            loss = round_stats["test_loss"]
            pbar.set_postfix({"Acc (%)": acc, "Loss": loss})

            if acc > best_accuracy:
                best_accuracy = acc
                best_metrics = round_stats

        # Final evaluation
        final_eval = evaluate_model(
            model=self.global_model,
            data_loader=self.test_loader,
            criterion=self.criterion,
            device=self.device
        )
        final_eval["best_round_accuracy"] = best_accuracy

        # Persist results
        summary_path = self.logger.finalize(final_eval)
        print(f"\n[FedShield Server] Training Complete! Final Test Accuracy: {final_eval['accuracy']}%")
        return {
            "final_metrics": final_eval,
            "experiment_id": self.logger.experiment_id,
            "log_path": summary_path,
            "rounds_history": self.logger.data["rounds"]
        }
