import torch

def compute_diagonal_sharded(C_shards, K_global, sigma, epsilon=1e-5):
    """
    Computes the diagonal preconditioner h = diag(B_t) for the ANP-Diagonal protocol.
    
    Paper Reference: Eq. (12) (Extraction) and Eq. (13) (Safety Clamping).
    
    Formula:
        h_j = (1/sigma) * (1 - c_j^T * K * c_j)
    
    Complexity:
        O(m * d/P) per shard. No All-Reduce required for the diagonal itself 
        (embarrassingly parallel after K is broadcasted).
        
    Args:
        C_shards (list[Tensor]): List of P sketch fragments [d_i, m].
        K_global (Tensor): The computed inverse kernel [m, m].
        sigma (float): Global damping parameter.
        epsilon (float): Minimum value for the preconditioner (numerical safety).
        
    Returns:
        h_shards (list[Tensor]): List of P diagonal fragments [d_i].
        stats (dict): Diagnostics about the raw values and clamping frequency.
    """
    h_sharded = []
    
    # Statistics accumulators
    total_elements = 0
    clamped_low = 0
    clamped_high = 0
    min_raw = float('inf')
    max_raw = float('-inf')
    
    upper_bound = 1.0 / sigma
    
    for C_i in C_shards:
        # -------------------------------------------------------------
        # 1. Efficient Diagonal Calculation (Eq. 12)
        # -------------------------------------------------------------
        # We need diag(C_i * K * C_i^T).
        # Let M = C_i * K  [shape: d_i, m]
        # Then (M * C_i^T)_jj = row_j(M) dot row_j(C_i)
        
        M_i = torch.mm(C_i, K_global)
        
        # Element-wise multiply and sum across columns (dim=1)
        # This gives the dot product for each row efficiently
        quad_form = (M_i * C_i).sum(dim=1) # [d_i]
        
        # h_raw = (1 - cKc^T) / sigma
        h_raw = (1.0 - quad_form) / sigma
        
        # -------------------------------------------------------------
        # 2. Collect Statistics (For "Platinum" Evidence)
        # -------------------------------------------------------------
        # Track raw statistics before clamping to prove why clamping is needed.
        # If min_raw < 0, it means the approximate Hessian was indefinite locally.
        # If max_raw > 1/sigma, it's mathematically impossible for SPD B_t, implies numerical error.
        with torch.no_grad():
            min_raw = min(min_raw, h_raw.min().item())
            max_raw = max(max_raw, h_raw.max().item())
            
            # Count violations
            clamped_low += (h_raw < epsilon).sum().item()
            clamped_high += (h_raw > upper_bound).sum().item()
            total_elements += h_raw.numel()
        
        # -------------------------------------------------------------
        # 3. Safety Clamping (Eq. 13)
        # -------------------------------------------------------------
        # Enforce spectral bounds: [epsilon, 1/sigma]
        # Note: In theoretical analysis, h_raw should be <= 1/sigma because 
        # cKc^T >= 0 for SPD K. However, FP8 noise might violate this.
        h_safe = h_raw.clamp(min=epsilon, max=upper_bound)
        
        h_sharded.append(h_safe)
        
    # Compile stats
    stats = {
        'min_raw_val': min_raw,
        'max_raw_val': max_raw,
        'clamp_rate_low': clamped_low / total_elements if total_elements > 0 else 0,
        'clamp_rate_high': clamped_high / total_elements if total_elements > 0 else 0,
        'total_clamp_rate': (clamped_low + clamped_high) / total_elements if total_elements > 0 else 0
    }
    
    return h_sharded, stats

def apply_adaptive_soft_threshold(u_shards, h_shards, lr, l1_lambda):
    """
    Applies the Consistent Metric Soft-Thresholding operator.
    
    Paper Reference: Eq. (15)
    
    x_{t+1} = sign(u) * max(0, |u| - threshold)
    Crucially, threshold = lr * lambda * h
    
    This ensures that the shrinkage is adapted to the local curvature h.
    High curvature (small h) -> small shrinkage (preserve feature).
    Low curvature (large h) -> large shrinkage (prune noise).
    
    Args:
        u_shards (list[Tensor]): Preconditioned gradient step (y_t - eta * h * grad).
        h_shards (list[Tensor]): Diagonal preconditioner.
        lr (float): Step size (eta).
        l1_lambda (float): Regularization strength.
    
    Returns:
        x_new_shards (list[Tensor]): Updated parameters.
    """
    x_new_shards = []
    
    for u_i, h_i in zip(u_shards, h_shards):
        # Adaptive Threshold
        thresh_i = lr * l1_lambda * h_i
        
        # Soft Thresholding Kernel
        # x = sign(u) * (abs(u) - thresh)+
        magnitude = u_i.abs() - thresh_i
        magnitude = magnitude.clamp(min=0.0)
        x_i = u_i.sign() * magnitude
        
        x_new_shards.append(x_i)
        
    return x_new_shards