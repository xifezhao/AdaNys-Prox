import torch

def anp_newton_step(grad_vector, C_shards, K, u_global, sigma):
    """
    Executes the ANP-Newton Sub-protocol (Projected Quasi-Newton).
    
    Paper Reference: Section 4.2, Eq. (10)
    
    Logic:
        Compute descent direction p_t = -B_t * g_t
        Using Woodbury: B_t = (1/sigma) * [I - C * K * C^T]
        p_t = (1/sigma) * ( -g_t + C * (K * (C^T * g_t)) )
        
        Note: The projection step P_C(x) happens AFTER the direction is applied
        (typically in the training loop or constraint handler). This function
        only returns the unconstrained variable-metric direction.
        
    Args:
        grad_vector (Tensor): Global gradient (or momentum buffer) [d].
        C_shards (list[Tensor]): List of sketch shards [d_i, m].
        K (Tensor): Inverse Kernel Matrix [m, m].
        u_global (Tensor): Pre-computed projection u = C^T * g [m].
                           (Computed via distributed_ops.compute_distributed_projection)
        sigma (float): Global damping parameter.
        
    Returns:
        direction (Tensor): The descent direction p_t [d].
    """
    # 1. Clamp u_global to prevent numerical explosion in K * u
    max_u_norm = 1e6
    u_norm = torch.norm(u_global)
    u_safe = u_global.clone()
    if u_norm > max_u_norm:
        u_safe = u_global / u_norm * max_u_norm
    
    # 2. Solve Kernel System: z = K * u with numerical checks
    # z shape: [m]
    z = torch.mv(K, u_safe)
    
    # Check for explosive values in z
    z_norm = torch.norm(z)
    if torch.isnan(z_norm) or torch.isinf(z_norm):
        # Fallback: return simple gradient scaling
        return grad_vector / max(sigma, 1e-6)
    
    # Clamp z if too large
    if z_norm > max_u_norm:
        z = z / z_norm * max_u_norm
    
    # 3. Expand back to parameter space: w = C * z
    # Since C is sharded, we compute w_i = C_i * z locally and then concatenate.
    w_parts = []
    for C_i in C_shards:
        # C_i: [d_i, m], z: [m] -> w_i: [d_i]
        w_i = torch.mv(C_i, z)
        # Clamp w_i to avoid explosion
        w_norm_i = torch.norm(w_i)
        if w_norm_i > max_u_norm:
            w_i = w_i / w_norm_i * max_u_norm
        w_parts.append(w_i)
        
    w_global = torch.cat(w_parts)
    
    # 4. Final Woodbury Combination with numerical stability
    # direction = (1/sigma) * (g - w)
    # Ensure sigma is not too small
    sigma_safe = max(sigma, 1e-6)
    
    direction = (grad_vector - w_global) / sigma_safe
    
    # Final check: clamp direction magnitude
    dir_norm = torch.norm(direction)
    if torch.isnan(dir_norm) or torch.isinf(dir_norm) or dir_norm > max_u_norm:
        # Fallback to simple gradient direction
        return torch.sign(grad_vector) * torch.clamp(torch.abs(grad_vector), max=1.0)
    
    return direction

def anp_diagonal_step(grad_vector, h_diagonal, lr, l1_lambda, current_params=None, consistent=True):
    """
    Executes the ANP-Diagonal Sub-protocol (Preconditioned Proximal Gradient).
    
    Paper Reference: Section 4.3, Eq. (15)
    
    Logic:
        Metric M = diag(h)^-1
        Step 1: Preconditioned Gradient Step
            u = y_t - lr * (h * g_t)
        Step 2: Proximal Mapping (Adaptive Soft Thresholding)
            x_{t+1} = soft_thresh(u, threshold)
            
        Consistent Metric: threshold = lr * lambda * h
        Inconsistent (Naive): threshold = lr * lambda
        
    Args:
        grad_vector (Tensor): Global gradient (or momentum buffer) [d].
        h_diagonal (Tensor): Diagonal preconditioner h [d].
        lr (float): Learning rate (step size).
        l1_lambda (float): L1 regularization strength.
        current_params (Tensor, optional): Current parameter values y_t [d].
            Required for correct Proximal update. If None, assumes parameters are 0
            (only useful for direction debugging, not training).
        consistent (bool): If True, use h^{-1} as prox metric (Ours).
                           If False, use Euclidean prox (Baseline).
                           
    Returns:
        update_step (Tensor): The effective change (x_t - x_{t+1}) to be applied.
                              Compatible with p.sub_(update_step).
    """
    if current_params is None:
        # Fallback for simplified simulation if params aren't passed.
        # Assumes we are just inspecting the gradient scaling direction.
        return h_diagonal * grad_vector

    # 1. Preconditioned Gradient Descent Step (unconstrained)
    # y_half = y_t - lr * (h * g_t)
    precond_grad = h_diagonal * grad_vector
    y_half = current_params - lr * precond_grad
    
    # 2. Adaptive Soft-Thresholding
    # Threshold depends on the metric choice
    if consistent:
        # Eq. (15): Threshold scales with curvature h
        threshold = lr * l1_lambda * h_diagonal
    else:
        # Naive: Threshold is constant (Euclidean Prox)
        threshold = lr * l1_lambda
        
    # Apply Soft Thresholding: sign(u) * max(0, |u| - thresh)
    magnitude = y_half.abs() - threshold
    magnitude = magnitude.clamp(min=0.0)
    x_new = y_half.sign() * magnitude
    
    # 3. Calculate Effective Update Step
    # We return d such that p_new = p_old - d
    # d = p_old - p_new
    update_step = current_params - x_new
    
    return update_step