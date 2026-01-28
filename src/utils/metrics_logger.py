import os
import json
import time
import logging
from datetime import datetime
import torch

logger = logging.getLogger(__name__)

class MetricsLogger:
    """
    Unified Logger for AdaNys-Prox Experiments.
    
    Responsibilities:
    1. Per-step logging of Loss, Gradient Norm, and Wall-clock time.
    2. Diagnostic logging: Jitter, Condition Numbers, Gram Errors.
    3. System logging: Simulated Communication Bytes, Memory Usage.
    4. Event logging: Trigger firings, Fallbacks.
    
    Format: JSON Lines (.jsonl) for robustness and flexibility with mixed types.
    """

    def __init__(self, output_dir, experiment_id, config=None):
        """
        Args:
            output_dir (str): Base directory for results.
            experiment_id (str): Unique identifier for the run.
            config (dict, optional): The full configuration dict to save as metadata.
        """
        self.output_dir = output_dir
        self.experiment_id = experiment_id
        
        # Create directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.output_dir, f"{experiment_id}_{timestamp}.jsonl")
        self.meta_file = os.path.join(self.output_dir, f"{experiment_id}_{timestamp}_meta.json")
        
        # Save Config Metadata immediately
        if config:
            with open(self.meta_file, 'w') as f:
                json.dump(config, f, indent=4)
        
        # Init Start Time
        self.start_time = time.time()
        
        logger.info(f"Logger initialized. Saving to: {self.log_file}")

    def log_step(self, step, loss, 
                 opt_metrics=None, 
                 sys_metrics=None, 
                 task_metrics=None,
                 grad_norm=0.0):
        """
        Records a single training step.
        
        Args:
            step (int): Global step count.
            loss (float): Current loss value.
            opt_metrics (dict): Metrics returned by AdaNysProx.step() 
                                (e.g., jitter_needed, trigger_fired, gram_error).
            sys_metrics (dict): Metrics from CommsModel 
                                (e.g., comm_bytes, simulated_time).
            task_metrics (dict): Task-specifics (e.g., sparsity F1, accuracy).
            grad_norm (float): Norm of the gradient (for convergence checks).
        """
        
        # Base record
        record = {
            'step': step,
            'loss': loss,
            'grad_norm': grad_norm,
            'wall_clock_real': time.time() - self.start_time,
            'timestamp': datetime.now().isoformat()
        }
        
        # Merge Optimizer Metrics (Diagnostics)
        if opt_metrics:
            # Flatten critical diagnostics for easy plotting later
            record.update({
                'jitter': opt_metrics.get('jitter_needed', 0.0),
                'cond_num': opt_metrics.get('condition_num', 0.0),
                'min_eig': opt_metrics.get('min_eig', 0.0),
                'gram_error': opt_metrics.get('gram_error', 0.0),
                'trigger_fired': opt_metrics.get('trigger_fired', False),
                'trigger_reason': opt_metrics.get('trigger_reason', 'none'),
                'diag_clamp_rate': opt_metrics.get('total_clamp_rate', 0.0),
                'is_cliff_hit': opt_metrics.get('is_cliff_hit', False)
            })

        # Merge System Metrics (Simulation)
        if sys_metrics:
            record.update({
                'simulated_time': sys_metrics.get('total_time', 0.0),
                'comm_bytes_update': sys_metrics.get('update_bytes', 0),
                'comm_bytes_reuse': sys_metrics.get('reuse_bytes', 0),
                'total_comm_bytes': sys_metrics.get('total_effective_bytes', 0)
            })
            
        # Merge Task Metrics (e.g., Sparsity)
        if task_metrics:
            record.update(task_metrics)

        # Write to disk (Flush every line to prevent data loss on crash)
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(record) + '\n')

    def log_event(self, event_type, message, details=None):
        """
        Logs a special sparse event (e.g., OOM warning, Fallback triggered).
        Saved to the same stream but with a special flag.
        """
        record = {
            'type': 'EVENT',
            'event': event_type,
            'message': message,
            'details': details or {},
            'wall_clock_real': time.time() - self.start_time
        }
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(record) + '\n')
    
    def log_memory_snapshot(self, label, sketches=None, accumulated=None, full_precision=None):
        """
        Logs memory usage for quantization validation.
        
        Args:
            label (str): Description (e.g., 'A1_FP32', 'A5_FP8')
            sketches (dict): Memory in {dtype: size_bytes}
            accumulated (dict): FP32 accumulator size
            full_precision (dict): Full precision parameter size
        """
        record = {
            'type': 'MEMORY',
            'label': label,
            'sketches_mb': {k: v / (1024**2) for k, v in sketches.items()} if sketches else {},
            'accumulated_mb': accumulated / (1024**2) if accumulated else 0,
            'full_precision_mb': full_precision / (1024**2) if full_precision else 0,
            'wall_clock_real': time.time() - self.start_time
        }
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(record) + '\n')
            
    def close(self):
        """Finalize logs."""
        pass # Nothing strictly needed for append-only file