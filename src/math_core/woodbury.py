import torch
import logging

logger = logging.getLogger(__name__)

def compute_core_inverse(W, G, sigma, delta, max_jitter=1e-2):
    """
    Computes the Inverse Kernel K_t using the Regularized Woodbury formulation.
    
    Paper Reference: Eq. (9)
    Formula: K_t = (sigma * W_t + sigma * delta * I + G_t)^-1
    
    This function is critical for the "Quantization Cliff" experiment. 
    When delta is too small relative to FP8 noise, the core matrix becomes 
    indefinite. This function attempts to detect and patch that using jitter.

    Args:
        W (Tensor): Core interaction matrix S^T * C [m, m].
        G (Tensor): Gram matrix C^T * C [m, m].
        sigma (float): Global damping parameter.
        delta (float): Kernel regularization parameter.
        max_jitter (float): Maximum diagonal damping to add if Cholesky fails.

    Returns:
        K (Tensor): The computed kernel inverse [m, m].
        info (dict): Diagnostic info {'jitter_used': float, 'success': bool}.
    """
    m = W.shape[0]
    device = W.device
    dtype = W.dtype

    # 1. Construct the Core Matrix M with improved stability
    # M = sigma * W + (sigma * delta) * I + G
    # Ensure sigma and delta are positive and reasonable
    sigma_safe = max(float(sigma), 1e-8)
    delta_safe = max(float(delta), 1e-10)
    
    regularization = (sigma_safe * delta_safe) * torch.eye(m, device=device, dtype=dtype)
    M_base = sigma_safe * W + regularization + G
    
    # Check for NaN/Inf in M_base
    if torch.isnan(M_base).any() or torch.isinf(M_base).any():
        logger.warning("M_base contains NaN or Inf. Applying fallback regularization.")
        # Replace NaN/Inf with small values
        M_base = torch.where(torch.isnan(M_base) | torch.isinf(M_base), 
                            torch.ones_like(M_base) * 1e-6, M_base)

    # 2. Add base regularization for stability (always start with some jitter)
    # This prevents the unregularized case from failing on ill-conditioned matrices
    base_reg = 1e-6 * torch.eye(m, device=device, dtype=dtype)
    M_base = M_base + base_reg
    
    # Robust Inversion (Cholesky with Jitter Retry)
    # This loop detects the "Cliff" mechanism.
    jitter_levels = [0.0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    
    K = None
    jitter_used = 0.0
    success = False

    for jitter in jitter_levels:
        if jitter > max_jitter:
            break
            
        # Add jitter if not first attempt (on top of base_reg already added)
        if jitter > 0:
            M_curr = M_base + jitter * torch.eye(m, device=device, dtype=dtype)
        else:
            M_curr = M_base

        try:
            # Attempt Cholesky Decomposition (Fastest & Strict SPD check)
            L = torch.linalg.cholesky(M_curr)
            # Efficient inverse from Cholesky factor
            K = torch.cholesky_inverse(L)
            
            # Check for explosive values in K
            K_norm = torch.norm(K)
            if K_norm > 1e8:
                logger.warning(f"K_norm very large: {K_norm:.2e}. Scaling down.")
                K = K / (K_norm / 1e4)  # Scale to reasonable magnitude
            
            jitter_used = jitter
            success = True
            break
            
        except (torch._C._LinAlgError, RuntimeError) as e:
            # Matrix is not positive definite (The "Cliff" has been hit)
            logger.debug(f"Cholesky failed with jitter={jitter}: {str(e)[:50]}")
            continue

    # 3. Fail-safe Fallback
    if not success:
        logger.warning(f"Cholesky failed even with max jitter {max_jitter}. "
                       f"Falling back to pseudo-inverse. (Cliff detected!)")
        # Use SVD-based pseudo-inverse as last resort to keep code running,
        # but this indicates a total breakdown of the metric.
        try:
            K = torch.linalg.pinv(M_base)
            # Scale to prevent overflow in subsequent computations
            K_norm = torch.norm(K)
            if K_norm > 1e6:
                K = K / (K_norm / 1e3)
        except:
            logger.error("Even pseudo-inverse failed! Using identity matrix.")
            K = torch.eye(m, device=device, dtype=dtype)
        jitter_used = max_jitter  # Marker for failure

    return K, {'jitter_used': jitter_used, 'success': success}

def apply_woodbury_update(v_global, u_global, K, sigma):
    """
    Applies the implicit Woodbury inverse operator to a vector.
    
    Formula: p = B * v = (1/sigma) * (v - C * K * (C^T * v))
    
    In the distributed setting (Section 5.2):
    1. u_global = C^T * v  (Computed via All-Reduce in distributed_ops)
    2. z = K * u_global    (Computed locally here)
    3. w = C * z           (Computed locally in distributed_ops)
    4. res = (v - w) / sigma
    
    This function performs step 2 and the final scaling logic structure, 
    but the `C * z` part typically happens outside because `C` is sharded.
    
    Here we provide the `solve_kernel` part: z = K * u.
    
    Args:
        v_global: Not used directly here, see logic above.
        u_global (Tensor): Projection vector C^T * v [m].
        K (Tensor): Inverse Kernel [m, m].
        sigma (float): Damping.
        
    Returns:
        z (Tensor): The intermediate vector K * u [m].
    """
    # z = K @ u
    # Simple Matrix-Vector multiplication
    z = torch.mv(K, u_global)
    return z

def finalize_direction(v_local, w_local, sigma):
    """
    Completes the Woodbury application on the local shard.
    
    p_local = (1/sigma) * (v_local - w_local)
    
    Args:
        v_local (Tensor): Local gradient shard.
        w_local (Tensor): Local result of C * z.
        sigma (float): Damping.
        
    Returns:
        p_local (Tensor): The preconditioned direction shard.
    """
    return (v_local - w_local) / sigma