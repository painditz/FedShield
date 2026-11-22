"""
Data Partitioner for Federated Learning in FedShield.
Supports:
1. IID (Independent and Identically Distributed) uniform partitioning.
2. Non-IID Dirichlet partitioning over class label distributions (alpha parameter).
3. Client sample quantity imbalance.
4. Profiling and inspection of per-client class distributions.
"""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import torch
from torch.utils.data import Dataset, Subset, DataLoader


class FederatedPartitioner:
    """
    Partitions a PyTorch classification dataset among a specified number of federated clients.
    """

    def __init__(
        self,
        dataset: Dataset,
        num_clients: int = 10,
        partition_type: str = "iid",
        dirichlet_alpha: float = 0.5,
        imbalance: bool = False,
        seed: int = 42
    ):
        self.dataset = dataset
        self.num_clients = num_clients
        self.partition_type = partition_type.lower()
        self.dirichlet_alpha = dirichlet_alpha
        self.imbalance = imbalance
        self.seed = seed

        # Extract targets
        if hasattr(dataset, "targets"):
            self.targets = np.array(dataset.targets)
        elif hasattr(dataset, "labels"):
            self.targets = np.array(dataset.labels)
        else:
            self.targets = np.array([dataset[i][1] for i in range(len(dataset))])

        self.num_classes = len(np.unique(self.targets))
        self.client_indices: List[List[int]] = []
        self._partition()

    def _partition(self) -> None:
        """Executes the chosen partitioning strategy."""
        np.random.seed(self.seed)
        num_samples = len(self.dataset)

        if self.partition_type == "iid":
            self.client_indices = self._partition_iid(num_samples)
        elif self.partition_type in ["non_iid", "non_iid_dirichlet", "dirichlet"]:
            self.client_indices = self._partition_dirichlet()
        else:
            raise ValueError(f"Unknown partition_type: {self.partition_type}. Choose 'iid' or 'non_iid_dirichlet'.")

    def _partition_iid(self, num_samples: int) -> List[List[int]]:
        """Randomly permutes indices and divides them uniformly among clients."""
        indices = np.random.permutation(num_samples)
        client_indices: List[List[int]] = []

        if self.imbalance:
            # Generate varying fractions summing to 1.0
            weights = np.random.uniform(0.5, 1.5, size=self.num_clients)
            proportions = weights / weights.sum()
            split_points = (np.cumsum(proportions) * num_samples).astype(int)[:-1]
            splits = np.split(indices, split_points)
            client_indices = [split.tolist() for split in splits]
        else:
            splits = np.array_split(indices, self.num_clients)
            client_indices = [split.tolist() for split in splits]

        return client_indices

    def _partition_dirichlet(self) -> List[List[int]]:
        """
        Partitions data using Dirichlet distribution over class distributions.
        For each class k, proportions p_{k, i} ~ Dir(alpha) determine the fraction of
        class k data allocated to client i.
        """
        client_indices: List[List[int]] = [[] for _ in range(self.num_clients)]

        for c in range(self.num_classes):
            class_indices = np.where(self.targets == c)[0]
            np.random.shuffle(class_indices)
            num_class_samples = len(class_indices)

            # Sample Dirichlet proportions for this class
            proportions = np.random.dirichlet(np.repeat(self.dirichlet_alpha, self.num_clients))

            if self.imbalance:
                # Modulate by client weights
                client_weights = np.random.uniform(0.6, 1.4, size=self.num_clients)
                proportions = proportions * client_weights
                proportions = proportions / proportions.sum()

            # Compute split points
            counts = (proportions * num_class_samples).astype(int)
            # Adjust rounding differences
            diff = num_class_samples - counts.sum()
            if diff > 0:
                counts[:diff] += 1
            elif diff < 0:
                counts[diff:] -= 1

            split_indices = np.split(class_indices, np.cumsum(counts)[:-1])
            for i in range(self.num_clients):
                client_indices[i].extend(split_indices[i].tolist())

        # Shuffle indices for each client
        for i in range(self.num_clients):
            np.random.shuffle(client_indices[i])

        return client_indices

    def get_client_dataset(self, client_id: int) -> Subset:
        """Returns PyTorch Subset dataset for given client."""
        if client_id < 0 or client_id >= self.num_clients:
            raise IndexError(f"client_id {client_id} out of range [0, {self.num_clients - 1}]")
        return Subset(self.dataset, self.client_indices[client_id])

    def get_client_dataloader(
        self,
        client_id: int,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0
    ) -> DataLoader:
        """Returns PyTorch DataLoader for a client."""
        client_dataset = self.get_client_dataset(client_id)
        return DataLoader(
            client_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )

    def get_client_class_distribution(self) -> np.ndarray:
        """
        Returns a 2D numpy array of shape (num_clients, num_classes)
        with sample count of class c at client i.
        """
        distribution = np.zeros((self.num_clients, self.num_classes), dtype=int)
        for i, indices in enumerate(self.client_indices):
            client_targets = self.targets[indices]
            for c in range(self.num_classes):
                distribution[i, c] = int(np.sum(client_targets == c))
        return distribution

    def get_partition_summary(self) -> Dict[str, Any]:
        """Provides summary statistics on client sample distributions."""
        sample_counts = [len(indices) for indices in self.client_indices]
        return {
            "num_clients": self.num_clients,
            "partition_type": self.partition_type,
            "dirichlet_alpha": self.dirichlet_alpha,
            "min_samples_per_client": int(min(sample_counts)),
            "max_samples_per_client": int(max(sample_counts)),
            "mean_samples_per_client": float(np.mean(sample_counts)),
            "std_samples_per_client": float(np.std(sample_counts)),
            "sample_counts": sample_counts,
            "class_distribution": self.get_client_class_distribution().tolist()
        }
