# 🛡️ FedShield: Federated Learning Security & Privacy Framework

**FedShield** is a comprehensive PyTorch framework for research and benchmarking in **Trustworthy Federated Learning (FL)**. It allows researchers and engineers to simulate distributed learning environments, evaluate optimization under extreme statistical data heterogeneity (Non-IID), benchmark Byzantine adversarial attacks, deploy robust aggregation defenses, and enforce differential privacy guarantees.

---

## 🌟 Key Features

1. **Distributed Optimization Algorithms**:
   - **FedAvg** (McMahan et al., 2017)
   - **FedProx** (Li et al., 2020) with proximal regularization parameter $\mu$ for system & statistical heterogeneity.
2. **Statistical Heterogeneity & Non-IID Partitioning**:
   - Class-based **Dirichlet Distribution** partitioning ($\alpha \in [0.05, 10.0]$).
   - Uniform IID partitioning and sample quantity imbalance simulations.
3. **Byzantine Adversarial Attacks**:
   - **Sign Flipping Attack**: Inverts the direction of client gradient updates ($-\text{scaling} \cdot \Delta w$).
   - **Update Scaling Attack**: Amplifies updates ($+\text{scaling} \cdot \Delta w$) to dominate aggregation.
   - **Random Noise & Zero Update Attacks**.
4. **Byzantine-Robust Server Aggregation**:
   - **Coordinate Median**: Dimension-wise median across clients.
   - **Trimmed Mean**: Eliminates extreme $\beta$-fraction parameter values per coordinate.
   - **Krum**: Pairwise Euclidean distance neighbor score optimization.
5. **Differential Privacy (DP)**:
   - $L_2$ norm update/gradient clipping threshold $C$.
   - Calibrated Gaussian noise addition.
   - Analytical Moments Accountant $(\epsilon, \delta)$-privacy tracking.
6. **Communication & Bandwidth Profiling**:
   - Analytical tracking of uplink and downlink communication overhead (Bytes, KB, MB, GB).
7. **Interactive Streamlit Web Dashboard**:
   - Real-time visualization of data heterogeneity, loss/accuracy convergence curves, defense comparisons, DP trade-offs, and an in-browser experiment launcher.

---

## 🏗️ Architecture

```
FedShield/
├── config/
│   ├── config.yaml          # Master configuration (dataset, model, federated, security, privacy)
│   └── config_parser.py     # Strongly-typed dataclass loader & device resolver
├── data/
│   ├── loader.py            # Automated CIFAR-10 downloader with anti-403 headers & augmentations
│   └── cifar-10-batches-py/ # Local CIFAR-10 data
├── models/
│   └── cnn.py               # Lightweight modular CNN with parameter vectorization & size estimators
├── clients/
│   ├── client.py            # Edge client representation with local SGD/FedProx/DP training
│   └── partitioner.py       # IID and Dirichlet Non-IID data partitioning engine
├── attacks/
│   └── poisoning.py         # Byzantine attacks (Sign-Flipping, Update-Scaling, Noise, Freeze)
├── server/
│   ├── server.py            # Federated Server orchestrator (rounds, client sampling, evaluation)
│   └── aggregation.py       # Robust aggregators (FedAvg, Coordinate Median, Trimmed Mean, Krum)
├── privacy/
│   └── differential_privacy.py # L2 clipping, Gaussian noise injection, Moments Accountant epsilon
├── algorithms/
│   └── federated_algorithm.py  # FedAvg and FedProx algorithm abstraction
├── evaluation/
│   ├── metrics.py           # Accuracy, F1, Loss, Precision, Recall, Confusion Matrix, Bandwidth
│   └── logger.py            # JSON detail logging and summary CSV updates
├── dashboard/
│   └── app.py               # Interactive multi-page Streamlit Dashboard
├── experiments/
│   ├── centralized.py       # Centralized benchmark baseline
│   ├── federated.py         # CLI-driven federated experiment runner
│   ├── run_benchmarks.py    # One-click comparative research benchmark suite
│   └── results/             # Saved experiment JSON files and summary CSV
└── tests/                   # 22 automated unit and integration tests
```

---

## ⚡ Quickstart Guide

### 1. Run the Test Suite (22/22 Passing)
```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

### 2. Launch the Web Dashboard Frontend
The project provides a lightweight, dependency-free interactive web frontend:

```powershell
.\.venv\Scripts\python.exe dashboard/web_app.py 8080
```
Then simply open **`http://localhost:8080`** in any web browser!

*(Optionally, you can also use Streamlit once installed: `.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py`)*

### 3. Run Centralized Benchmark Baseline
```powershell
.\.venv\Scripts\python.exe experiments/centralized.py --mode quick
```

### 4. Run Federated Learning Experiments

#### A. Standard FedAvg (IID)
```powershell
.\.venv\Scripts\python.exe experiments/federated.py --algorithm fedavg --partition iid --rounds 5 --clients 10 --mode quick
```

#### B. FedProx on Extreme Non-IID ($\alpha = 0.1$)
```powershell
.\.venv\Scripts\python.exe experiments/federated.py --algorithm fedprox --partition non_iid_dirichlet --alpha 0.1 --rounds 5 --mode quick
```

#### C. Byzantine Attack vs Robust Aggregation (Sign Flipping vs Trimmed Mean)
```powershell
# Attack on standard vulnerable FedAvg
.\.venv\Scripts\python.exe experiments/federated.py --algorithm fedavg --malicious-fraction 0.2 --attack sign_flipping --aggregation fedavg --mode quick

# Defense with Trimmed Mean
.\.venv\Scripts\python.exe experiments/federated.py --algorithm fedavg --malicious-fraction 0.2 --attack sign_flipping --aggregation trimmed_mean --mode quick

# Defense with Coordinate Median
.\.venv\Scripts\python.exe experiments/federated.py --algorithm fedavg --malicious-fraction 0.2 --attack sign_flipping --aggregation coordinate_median --mode quick
```

#### D. Differential Privacy (DP-FedAvg)
```powershell
.\.venv\Scripts\python.exe experiments/federated.py --algorithm fedavg --dp --noise-multiplier 0.5 --mode quick
```

---

## 📊 Automated Benchmark Suite

Run the full comparative research benchmark suite in a single command:
```powershell
.\.venv\Scripts\python.exe experiments/run_benchmarks.py --mode quick
```
This automatically runs and records:
1. **Benchmark A**: IID vs Non-IID Dirichlet data distribution impact on convergence.
2. **Benchmark B**: FedAvg vs FedProx performance under extreme class skew ($\alpha=0.1$).
3. **Benchmark C**: Byzantine sign-flipping attack vulnerability vs Trimmed Mean & Median defenses.
4. **Benchmark D**: Differential privacy noise-utility trade-off.

All results are automatically persisted to `experiments/results/experiments_summary.csv` and rendered dynamically in the dashboard.

---

## 🎤 Presentation Talking Points (For Your Demo)

1. **Problem Statement**: Standard machine learning requires centralizing sensitive user data. Federated learning trains models collaboratively across devices without exposing raw data.
2. **Key Challenge 1 - Data Heterogeneity**: Client data is non-IID. We demonstrate Dirichlet distribution partitioning ($\alpha=0.5$ vs $\alpha=0.1$) and show how **FedProx** prevents client drift using proximal regularization.
3. **Key Challenge 2 - Byzantine Poisoning**: Malicious edge participants can upload poisoned updates. We demonstrate **Sign Flipping** and **Update Scaling** attacks that drop FedAvg accuracy, and prove how **Coordinate Median** and **Trimmed Mean** neutralize the attacks.
4. **Key Challenge 3 - Privacy Leakage**: Even without sharing raw data, gradients can leak information. We implement $(\epsilon, \delta)$-**Differential Privacy** with $L_2$ clipping and calibrated Gaussian noise, showing formal privacy budget tracking.
5. **Interactive Dashboard**: Show the Streamlit UI with the live Dirichlet distribution heatmap, multi-run convergence comparison, and communication bandwidth profiler.
