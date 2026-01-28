import os
import yaml
import torch
import argparse
import logging
import copy
from datetime import datetime

# --- Import Internal Modules ---
from src.tasks.dense_quadratic import DenseQuadraticTask
from src.tasks.sparse_logistic import SparseLogisticTask
from src.optim.adanys_prox import AdaNysProx
from src.utils.metrics_logger import MetricsLogger

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [MATRIX] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==============================================================================
# 1. The Execution Engine (Single Run Logic)
# ==============================================================================
def run_single_experiment(config, run_id_suffix="", device="mps"):
    """
    Executes a single training configuration end-to-end.
    """
    # 1. Setup Config & ID
    exp_id = config['experiment']['id'] + run_id_suffix
    output_dir = config['experiment']['output_dir']
    seed = config['experiment']['seed']
    
    logger.info(f"STARTING: {exp_id} on {device}")
    torch.manual_seed(seed)
    
    # 2. Initialize Task (Data)
    task_conf = config['task']
    if task_conf['name'] == 'dense_quadratic':
        task = DenseQuadraticTask(
            dim=task_conf['dim'],
            condition_number=task_conf.get('condition_number', 1e6),
            spectrum=task_conf.get('spectrum_type', 'spiked'),
            data_path=task_conf['data_path'],
            device=device
        )
        hvp_oracle = task.hvp
    elif task_conf['name'] == 'sparse_logistic':
        task = SparseLogisticTask(
            dim=task_conf['dim'],
            sparsity=task_conf.get('sparsity', 0.01),
            data_path=task_conf['data_path'],
            device=device
        )
        hvp_oracle = task.hvp
    else:
        raise ValueError(f"Unknown task: {task_conf['name']}")

    # 3. Initialize Optimizer (AdaNys or Baseline)
    opt_conf = config['optimizer']
    adanys_conf = config.get('adanys', {'enabled': False})
    
    # Flatten parameters
    # Note: In simulation tasks, we usually optimize a single weight vector.
    # We create a parameter tensor from scratch or use task initialization if provided.
    # For synthetic tasks, we typically start from zero or random.
    w = torch.zeros(task.dim, device=device, requires_grad=True)
    
    if opt_conf['name'] == 'adanys_prox' and adanys_conf['enabled']:
        optimizer = AdaNysProx(
            [w],
            lr=opt_conf['params']['lr'],
            betas=tuple(opt_conf['params'].get('betas', (0.9, 0.999))),
            protocol=adanys_conf['protocol'],
            sketch_size_m=adanys_conf['sketch_size_m'],
            sigma=adanys_conf['sigma'],
            delta=adanys_conf['delta'],
            l1_lambda=0.0, # Can be overridden for sparse tasks
            # Sub-configs
            laziness_config=adanys_conf['laziness'],
            quant_config=adanys_conf['quantization'],
            cluster_config=config['system']['virtual_cluster']
        )
    elif opt_conf['name'] == 'adamw':
        optimizer = torch.optim.AdamW(
            [w], 
            lr=opt_conf['params']['lr'],
            weight_decay=opt_conf['params'].get('weight_decay', 0.0)
        )
    elif opt_conf['name'] == 'sgd':
        optimizer = torch.optim.SGD(
            [w],
            lr=opt_conf['params']['lr'],
            momentum=opt_conf['params'].get('momentum', 0.0)
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_conf['name']}")

    # 4. Initialize Logger
    metrics_logger = MetricsLogger(output_dir, exp_id, config)

    # 5. Training Loop
    max_steps = config['training']['max_steps']
    log_interval = config['training']['log_interval']
    
    try:
        for step in range(max_steps):
            def closure():
                optimizer.zero_grad()
                loss, grad = task.closure(w)
                # Manually set grad (since tasks compute it explicitly)
                w.grad = grad
                return loss

            # Optimizer Step
            # For AdaNys, we pass the explicit HVP oracle
            if isinstance(optimizer, AdaNysProx):
                loss, opt_metrics = optimizer.step(closure, hvp_oracle=hvp_oracle)
                
                # Get System Metrics
                sys_metrics = None
                if optimizer.cluster.comms:
                    sys_metrics = optimizer.cluster.comms.stats
            else:
                # Baseline step
                loss = closure()
                optimizer.step()
                opt_metrics = {} # No internal diagnostics for AdamW
                
                # Simulate O(d) comms for baseline
                sys_metrics = {}
                if 'virtual_cluster' in config['system']:
                    # Simple accumulation of time for comparison
                    # d * 4 bytes * 2 (Ring) / BW
                    d = task.dim
                    payload = d * 4
                    # Calculate theoretical time based on config params
                    bw = config['system']['virtual_cluster']['bandwidth_gbps'] * 1e9 / 8
                    lat = config['system']['virtual_cluster']['latency_us'] * 1e-6
                    t_comm = lat + (2 * payload / bw)
                    # Accumulate (linear approximation for baseline log)
                    sys_metrics['total_time'] = (step + 1) * t_comm 
                    sys_metrics['total_effective_bytes'] = (step + 1) * payload * 2

            # Task Metrics (e.g. Sparsity F1)
            task_metrics = {}
            if hasattr(task, 'get_support_metrics'):
                task_metrics = task.get_support_metrics(w)

            # Log
            if step % log_interval == 0:
                grad_norm = w.grad.norm().item()
                metrics_logger.log_step(
                    step, loss.item(), 
                    opt_metrics, sys_metrics, task_metrics, grad_norm
                )
                
                # Console feedback
                if step % (log_interval * 10) == 0:
                    cliff_warn = " [CLIFF]" if opt_metrics.get('is_cliff_hit') else ""
                    print(f"  Step {step}: Loss {loss.item():.6f}{cliff_warn}")

    except Exception as e:
        logger.error(f"Experiment FAILED: {str(e)}")
        metrics_logger.log_event("FAILURE", str(e))
        # Don't crash the matrix, just this run
    finally:
        metrics_logger.close()
        logger.info(f"FINISHED: {exp_id}\n")


# ==============================================================================
# 2. The Matrix Orchestrator
# ==============================================================================
def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def run_matrix(args):
    """
    Decides which experiments to run and applies Parameter Sweeps.
    """
    
    # --- Experiment Group A0: Baselines ---
    if 'A0' in args.experiments:
        cfg = load_config('configs/A0_baseline.yaml')
        run_single_experiment(cfg, device=args.device)

    # --- Experiment Group A1: Ideal FP32 ---
    if 'A1' in args.experiments:
        cfg = load_config('configs/A1_nys_fp32.yaml')
        run_single_experiment(cfg, device=args.device)

    # --- Experiment Group A3: Stability Sweep (Tau) ---
    # Compare A2 (Unstable) logic vs A3 (Stable) logic over different taus
    if 'A3' in args.experiments:
        base_cfg = load_config('configs/A3_lazy_stable.yaml')
        
        # Sweep Tau
        for tau in [10, 20, 50, 100]:
            # 1. Run with Triggers (Stable)
            cfg_stable = copy.deepcopy(base_cfg)
            cfg_stable['adanys']['laziness']['tau_max'] = tau
            cfg_stable['adanys']['laziness']['tau_initial'] = tau
            run_single_experiment(cfg_stable, f"_Tau{tau}_TriggersON", device=args.device)
            
            # 2. Run without Triggers (Unstable - A2 logic)
            cfg_unstable = copy.deepcopy(base_cfg)
            cfg_unstable['adanys']['laziness']['tau_max'] = tau
            cfg_unstable['adanys']['laziness']['tau_initial'] = tau
            cfg_unstable['adanys']['laziness']['use_triggers'] = False
            cfg_unstable['experiment']['id'] = "A2_Lazy_Unstable" # Rename to match paper logic
            run_single_experiment(cfg_unstable, f"_Tau{tau}_TriggersOFF", device=args.device)

    # --- Experiment Group A5: Quantization Cliff (Delta Sweep) ---
    # This is the "Platinum" numerical proof.
    if 'A5' in args.experiments:
        base_cfg = load_config('configs/A5_fp8_cliff.yaml')
        
        # Logarithmic sweep crossing the FP8 noise floor (~1e-2 to 1e-3)
        # We go deep to 1e-9 to ensure we hit the cliff.
        deltas = [1e-2, 1e-3, 1e-4, 1e-6, 1e-7, 1e-8, 1e-9]
        
        for d in deltas:
            cfg = copy.deepcopy(base_cfg)
            cfg['adanys']['delta'] = float(d)
            
            # Format ID for clean sorting
            d_str = f"{d:.0e}".replace('-', 'n')
            run_single_experiment(cfg, f"_Delta{d_str}", device=args.device)

    # --- Experiment Group Consistency: Metric Check ---
    if 'Consistency' in args.experiments:
        # Load a base sparse config (assuming it exists or reuse logic)
        # Here we programmatically modify A3 or assume criteo_simulation.yaml exists
        # For brevity, I'll assume we adapt A3 to sparse logic
        base_cfg = load_config('configs/A3_lazy_stable.yaml')
        
        # Convert to Sparse Task
        base_cfg['task']['name'] = 'sparse_logistic'
        base_cfg['task']['data_path'] = 'data/sparse_d50k.pt'
        base_cfg['task']['dim'] = 50000
        base_cfg['adanys']['protocol'] = 'diagonal'
        base_cfg['optimizer']['params']['l1_lambda'] = 1e-4 # Add regularization
        
        # Run 1: Consistent (Ours)
        cfg_ours = copy.deepcopy(base_cfg)
        cfg_ours['experiment']['id'] = "Metric_Consistent"
        # Logic for consistent is default in subprotocols.py if not specified
        run_single_experiment(cfg_ours, "_Ours", device=args.device)
        
        # Run 2: Inconsistent (Naive)
        # Note: Ideally, subprotocols.py should accept a flag passed from here.
        # Since our config schema is simple, we might need to rely on the implementation 
        # defaulting to consistent, and maybe we hack/mock for the baseline, 
        # OR we just explicitly compare against SGD+L1.
        # For Platinum rigor, let's compare against a "Naive Nystrom" which uses B*g but Euclidean Prox.
        # This would require code support in subprotocols.py (added 'consistent' arg).
        pass 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AdaNys-Prox M1 Experiment Matrix")
    
    parser.add_argument("--device", type=str, default="mps", 
                        help="Device to run simulation (mps/cpu)")
    
    parser.add_argument("--experiments", nargs="+", default=["A0", "A1", "A3", "A5"],
                        choices=["A0", "A1", "A3", "A5", "Consistency"],
                        help="List of experiment groups to run")
    
    args = parser.parse_args()
    
    print("="*60)
    print(f"🚀 Launching AdaNys-Prox Matrix on {args.device.upper()}")
    print(f"📋 Selected Experiments: {args.experiments}")
    print("="*60)
    
    run_matrix(args)