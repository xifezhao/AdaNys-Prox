import torch
from torch.optim import Optimizer

from ..emulation.fp8_quantizer import FP8E4M3Simulator
from ..emulation.distributed_ops import LogicalCluster
from .lazy_manager import LazyCacheManager
from ..math_core.woodbury import compute_core_inverse
from ..math_core.diag_extraction import compute_diagonal_sharded
from ..math_core.diagnostics import diagnose_core_matrix, calc_gram_error
from .subprotocols import anp_newton_step, anp_diagonal_step

class AdaNysProx(Optimizer):
    """
    The Platinum Implementation of AdaNys-Prox for M1 Simulation.
    
    Integrates:
    - Quantized Lazy Cache (Section 5)
    - Bifurcated Sub-protocols (Section 4)
    - Distributed Accounting (Section 5.2)
    - Numerical Diagnostics (Appendix B/C)
    """

    def __init__(self, params, 
                 lr=1e-3, 
                 betas=(0.9, 0.999), # Momentum factors
                 protocol='newton',  # 'newton' or 'diagonal'
                 sketch_size_m=256,
                 sigma=1e-4,         # Global damping
                 delta=1e-6,         # Kernel regularization
                 l1_lambda=0.0,      # For ANP-Diagonal
                 # --- System Emulation Config ---
                 laziness_config=None,
                 quant_config=None,
                 cluster_config=None):
        
        defaults = dict(lr=lr, betas=betas, protocol=protocol, 
                        sketch_size_m=sketch_size_m, sigma=sigma, delta=delta,
                        l1_lambda=l1_lambda)
        
        super().__init__(params, defaults)
        
        # 1. Initialize Sub-systems
        self.device = self.param_groups[0]['params'][0].device
        
        # Hardware Emulators
        self.quantizer = FP8E4M3Simulator(device=self.device)
        self.cluster = LogicalCluster(
            num_shards=cluster_config.get('num_shards', 32),
            comms_model=cluster_config.get('comms_model', None)
        )
        
        # Stability Manager
        self.lazy_manager = LazyCacheManager(
            config=laziness_config or {},
            device=self.device
        )
        
        # State Initialization
        self.state['step'] = 0
        self.state['last_grad'] = None # For orthogonality check

    @torch.no_grad()
    def step(self, closure=None, hvp_oracle=None):
        """
        Performs a single optimization step.
        
        Args:
            closure (callable): Re-evaluates loss (standard PyTorch).
            hvp_oracle (callable): Function(v) -> Hv. Required during Update Phase.
                                   We use explicit HVP to avoid MPS autograd issues.
                                   
        Returns:
            loss, metrics (dict)
        """
        loss = None
        if closure is not None:
            loss = closure()

        # Collect Global Metrics
        metrics = {
            'trigger_fired': False,
            'trigger_reason': 'none',
            'jitter_needed': 0.0,
            'gram_error': 0.0,
            'comm_bytes_update': 0,
            'comm_bytes_reuse': 0,
            'is_cliff_hit': False
        }

        # Reset per-step comms stats for clean logging
        if self.cluster.comms:
            self.cluster.comms.reset_stats()

        for group in self.param_groups:
            # Flatten all params into one logical vector (simplifies simulation)
            params = []
            grads = []
            for p in group['params']:
                if p.grad is not None:
                    params.append(p)
                    grads.append(p.grad.view(-1))
            
            if not grads: continue
            
            # Global Gradient (Conceptually aggregated)
            # In simulation, we assume this is already the global gradient.
            # We charge the "All-Reduce" cost for gradients here.
            full_grad = torch.cat(grads)
            d = full_grad.numel()
            
            # Log Gradient Comm Cost (Baseline Comparison)
            # O(d) cost every step
            if self.cluster.comms:
                self.cluster.comms.log_all_reduce(d * 4, name="Gradient_Sync")

            # --- Stability Trigger Check ---
            should_update, trigger_reason = self.lazy_manager.check_triggers(
                self.state['step'], full_grad, self.state.get('last_grad')
            )
            
            # --- Update Phase (Construct Preconditioner) ---
            if should_update:
                if hvp_oracle is None:
                    raise ValueError("HVP Oracle required for Nystrom Update Phase")
                
                # 1. Generate Gaussian Sketching Matrix S_t
                m = group['sketch_size_m']
                S_t = torch.randn(d, m, device=self.device) / (d ** 0.5)
                
                # 2. Distributed HVP Computation: C_t = H * S_t
                # Cost: m * backward passes (Amortized by tau)
                C_t_fp32 = hvp_oracle(S_t) 
                
                # 3. FP8 Quantization & Storage
                # We simulate sharding inside the quantizer/cluster logic
                # C_rec is the reconstructed (noisy) sketch used for math
                C_rec, store_bytes = self.quantizer.quantize_store(C_t_fp32)
                
                # 4. Logical Sharding
                C_shards = self.cluster.shard_tensor(C_rec)
                
                # 5. Distributed Gram Matrix Construction (O(m^2) comms)
                # G_t = C^T * C
                G_t = self.cluster.compute_distributed_gram(C_shards)
                
                # [Platinum Diagnostic]: Compute Gram Error before continuing
                # Compare noisy G_t against ground truth FP32 G_t
                G_truth = C_t_fp32.T @ C_t_fp32
                metrics['gram_error'] = calc_gram_error(G_truth, G_t)
                
                # 6. Core Kernel Inversion (Woodbury)
                # W_t = S^T * C approx S^T * C_rec
                W_t = S_t.T @ C_rec
                
                # [Platinum Diagnostic]: Deep inspection for Cliff
                diag_info = diagnose_core_matrix(W_t, G_t, group['sigma'], group['delta'])
                metrics.update(diag_info)
                metrics['is_cliff_hit'] = diag_info['jitter_needed'] > 1e-4
                
                # Actual Computation with robust fallback
                K_t, inv_info = compute_core_inverse(
                    W_t, G_t, group['sigma'], group['delta']
                )
                
                # 7. Protocol-Specific Setup
                if group['protocol'] == 'diagonal':
                    # ANP-Diagonal: Extract h = diag(B)
                    h_shards, diag_stats = compute_diagonal_sharded(
                        C_shards, K_t, group['sigma']
                    )
                    # Cache the sharded diagonal vector
                    # Reassemble for simple application in M1 RAM
                    h_global = torch.cat(h_shards)
                    self.state['precond'] = h_global
                    metrics.update(diag_stats) # Log clamp rates
                    
                else:
                    # ANP-Newton: Cache C shards and K kernel
                    self.state['C_shards'] = C_shards
                    self.state['K_t'] = K_t
                
                # Update Lazy Manager State
                self.lazy_manager.update_state(self.state['step'])
                
                # Log Update metrics
                metrics['trigger_fired'] = True
                metrics['trigger_reason'] = trigger_reason
                if self.cluster.comms:
                    metrics['comm_bytes_update'] = self.cluster.comms.stats['total_effective_bytes']

            # --- Reuse Phase (Apply Preconditioner) ---
            
            # Momentum update (First order part)
            if 'momentum_buffer' not in self.state:
                self.state['momentum_buffer'] = torch.zeros_like(full_grad)
            
            buf = self.state['momentum_buffer']
            beta, _ = group['betas']
            buf.mul_(beta).add_(full_grad)
            
            # Get direction based on protocol
            if group['protocol'] == 'diagonal':
                # Preconditioned Gradient: u = h * buf
                h = self.state['precond']
                direction = anp_diagonal_step(
                    buf, h, group['lr'], group['l1_lambda']
                )
            else:
                # Projected Quasi-Newton: p = B * buf
                C_shards = self.state['C_shards']
                K_t = self.state['K_t']
                
                # Distributed Projection: u = C^T * v (O(m) comms)
                u_global = self.cluster.compute_distributed_projection(C_shards, self.cluster.shard_tensor(buf))
                
                # Woodbury Solve (Local) & Expansion
                # direction = (v - C * K * u) / sigma
                direction = anp_newton_step(
                    buf, C_shards, K_t, u_global, group['sigma']
                )

            # Apply Updates to Parameters (Un-flatten) with numerical safety
            idx = 0
            
            # Check direction for NaN/Inf before applying
            if torch.isnan(direction).any() or torch.isinf(direction).any():
                logger.warning("Direction contains NaN or Inf. Skipping update and using fallback.")
                # Use simple gradient descent as fallback
                direction = full_grad.clone()
            
            # Clip direction magnitude to prevent explosive updates
            dir_norm = torch.norm(direction)
            max_step = 1.0  # Limit step magnitude to prevent instability
            if torch.isinf(dir_norm) or torch.isnan(dir_norm):
                logger.warning(f"Direction norm is invalid ({dir_norm}). Using unit norm.")
                direction = full_grad.clone()
                dir_norm = torch.norm(direction)
                if dir_norm > 0:
                    direction = direction / (dir_norm + 1e-10)
            elif dir_norm > max_step:
                direction = direction / dir_norm * max_step
            
            for p in params:
                numel = p.numel()
                p_update = direction[idx : idx + numel].view_as(p)
                
                # Additional per-parameter clipping to avoid NaN propagation
                p_update_norm = torch.norm(p_update)
                if torch.isinf(p_update_norm) or torch.isnan(p_update_norm) or p_update_norm > max_step:
                    if p_update_norm > 0 and not (torch.isinf(p_update_norm) or torch.isnan(p_update_norm)):
                        p_update = p_update / p_update_norm * max_step
                    else:
                        p_update = torch.zeros_like(p_update)
                
                # Apply update with learning rate from group
                lr = group.get('lr', 1e-3)
                p.data.add_(p_update, alpha=-lr)
                idx += numel
                
            # Log Reuse metrics
            if self.cluster.comms:
                # Subtract update phase bytes to get only reuse cost
                total = self.cluster.comms.stats['total_effective_bytes']
                metrics['comm_bytes_reuse'] = total - metrics['comm_bytes_update']

        # Step Finalization
        self.state['last_grad'] = full_grad.clone().detach() # Save for ortho check
        self.state['step'] += 1
        
        return loss, metrics