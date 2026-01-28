# AdaNys-Prox M1 Lab: Logical Emulator

**Scalable Composite Optimization via Quantized Lazy Nyström Preconditioning**

> **Note on Hardware Environment**: This repository is a **Logical Emulator** designed for macOS Apple Silicon (M1/M2/M3). It does NOT implement high-performance CUDA kernels. Instead, it simulates the numerical behavior (FP8 quantization noise) and system constraints (bandwidth latency, memory limits) of a distributed H100 cluster to verify the algorithmic claims of the AdaNys-Prox paper.

---

## 🔬 The Philosophy of Logical Emulation

How do we verify a billion-scale distributed algorithm on a laptop? We decouple **logic** from **execution**.

1.  **FP8 Simulation (Numerical Rigor)**: 
    Since M1 `mps` backends lack native E4M3 tensor cores, we simulate FP8 behavior using FP32 math. We implement a **Discrete Lattice Model** that replicates the limited dynamic range (saturation at $\pm 448$) and relative precision (3-bit mantissa rounding) of the E4M3 format. This allows us to observe numerical phenomena like the "Quantization Cliff" and "Noise Floor".

2.  **Distributed Accounting (System Rigor)**:
    We physically execute computations on a single GPU (or CPU) but logically shard tensors into $P=32$ chunks. We intercept all cross-shard operations (e.g., constructing the Gram matrix) and calculate the **Theoretical Latency** based on a Ring All-Reduce model over 400Gbps InfiniBand. This proves the $O(d) \to O(m^2)$ communication reduction.

3.  **Explicit HVP (Algorithmic Rigor)**:
    We use synthetic tasks with explicit Hessian structures (e.g., Spiked Spectrum) to calculate Hessian-Vector Products directly. This avoids the black-box nature of Autograd and allows us to inject precise numerical perturbations.

---

## 🧪 Experiment Matrix (A0 - A5)

The repository is structured around a "Falsifiable Experiment Matrix". Each configuration targets a specific claim in the paper.

| ID | Config File | Hypothesis / Claim | Expected Phenomenon on M1 |
| :--- | :--- | :--- | :--- |
| **A0** | `A0_baseline.yaml` | **Baseline**: First-order methods are slow on ill-conditioned problems but cheap per step. | Slow convergence rate per step; high simulated comm cost ($O(d)$). |
| **A1** | `A1_nys_fp32.yaml` | **Ideal Upper Bound**: Nyström works perfectly with infinite precision and frequent updates. | Fastest convergence per step; slowest wall-clock time (due to freq. comms). |
| **A2** | `A2_lazy_unstable.yaml` | **The Risk**: Lazy updates without safeguards lead to drift. | **Loss Spikes** appear when $\tau=50$ under stochastic noise. |
| **A3** | `A3_lazy_stable.yaml` | **The Solution**: Stability Triggers detect drift and repair the metric. | Spikes are eliminated; logs show `Trigger Fired: Orthogonality/Fail`. |
| **A4** | `A4_fp8_safe.yaml` | **The Reality**: FP8 introduces a noise floor but enables convergence if $\delta$ is safe. | Loss converges initially like A1, then plateaus at a **Noise Floor** ($\approx 10^{-5}$). |
| **A5** | `A5_fp8_cliff.yaml` | **The Cliff**: If $\delta$ is too small, core inversion amplifies FP8 noise. | **Loss Divergence** or Cholesky Failure when $\delta < 10^{-7}$. `jitter_needed` spikes. |

---

## ⚙️ Technical Implementation Details

### 1. FP8 E4M3 Simulator
Located in `src/emulation/fp8_quantizer.py`.
We do **not** simply inject additive Gaussian noise. We implement a rigorous rounding scheme:
*   **Column-wise Scaling**: $x_{norm} = x / \max(|x|)$.
*   **Relative Stepping**: The grid step size $\Delta$ depends on the magnitude $|x_{norm}|$, simulating floating-point behavior.
*   **Absolute Flush**: Values $|x| < 2^{-9}$ are hard-flushed to zero (Subnormal handling).
*   **Saturation**: Values $|x| > 448$ are clamped.

### 2. Distributed Communication Model
Located in `src/emulation/comms_model.py`.
We model the time cost $T$ of synchronization using the Alpha-Beta model for Ring All-Reduce:
$$ T = \alpha + \beta \times \text{Effective\_Bytes} $$
*   $\alpha$ (Latency): 50 $\mu s$
*   $\beta$ (Bandwidth): Inverse of 400 Gbps.
*   **Effective Bytes**: $2 \times \frac{P-1}{P} \times \text{Payload}$.

### 3. Stability Triggers
Located in `src/optim/lazy_manager.py`.
We monitor three signals to interrupt the "Lazy" phase:
1.  **Max Age**: Hard limit ($\tau_{\max}=50$).
2.  **Orthogonality**: $\cos(\nabla f_t, \nabla f_{t-1}) < 0.05$. (Geometry Shift)
3.  **Line Search Failure**: Consecutive backtracks > 2. (Metric Mismatch)

---

## 🚀 Quick Start

### 1. Installation
Requires Python 3.9+ and PyTorch (MPS support recommended for speed, CPU works too).

```bash
pip install -r requirements.txt
```

### 2. Data Generation
Generate synthetic datasets with specific spectral properties to ensure reproducibility.

```bash
# Dense Quadratic with Spiked Spectrum (for Cliff/Stability tests)
python src/tasks/dense_quadratic.py --dim 2000 --cond 1e6 --spectrum spiked --out data/dense_d2000_cond1e6.pt

# Sparse Logistic Regression (for Metric Consistency tests)
python src/tasks/sparse_logistic.py --dim 50000 --sparsity 0.01 --out data/sparse_d50k.pt
```

### 3. Run the Experiment Matrix
Execute the full suite of experiments. This will perform parameter sweeps and log metrics to `results/metrics/`.

```bash
# Run the Quantization Cliff verification (Sweeps delta)
python run_matrix.py --experiments A5 --device mps

# Run the Stability verification (Sweeps tau, compares Trigger ON/OFF)
python run_matrix.py --experiments A3 --device mps

# Run Baseline comparison
python run_matrix.py --experiments A0 --device mps
```

### 4. Generate Report
Parse the logs and generate the PDF figures (saved to `results/figures/`).

```bash
python analyze_results.py
```

### 5. Verify Results
Check `results/figures/` for:
*   `Figure2_Quantization_Cliff.pdf`: Should show Jitter spiking as $\delta$ drops.
*   `Figure3_Stability_Triggers.pdf`: Should show A3 smoothing out A2's spikes.
*   `Figure1_System_Efficiency.pdf`: Should show massive reduction in communication volume.

---

## 📂 Directory Structure

```text
├── configs/            # YAML configurations for A0-A5
├── src/
│   ├── emulation/      # FP8 & Network simulators
│   ├── math_core/      # Woodbury, Diag extraction, Jitter diagnostics
│   ├── optim/          # AdaNysProx implementation
│   └── tasks/          # Synthetic problem definitions
├── results/            # Logs and Plots
└── run_matrix.py       # Main execution script
```

## Citation

If you use this codebase, please cite the associated ICML 2026 submission:
"AdaNys-Prox: Scalable Composite Optimization via Quantized Lazy Nyström Preconditioning".
