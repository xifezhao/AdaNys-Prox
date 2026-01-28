import torch
import logging

# Configure logger to capture numerical warnings
logger = logging.getLogger(__name__)

def robust_cholesky_with_jitter(M, max_jitter=1e-1):
    """
    Attempts Cholesky decomposition. If it fails (due to indefiniteness),
    iteratively adds diagonal jitter until it succeeds or gives up.
    
    This serves as a critical metric: 'jitter_needed'.
    - If 0.0: The system is healthy and naturally SPD.
    - If > 0.0: The system hit the 'Quantization Cliff' and required patching.
    
    Args:
        M (Tensor): Symmetric matrix [m, m].
        max_jitter (float): Upper bound for jitter injection.
        
    Returns:
        L (Tensor): Cholesky factor (lower triangular).
        jitter_used (float): The amount of damping added.
        success (bool): Whether decomposition succeeded eventually.
    """
    # Jitter schedule: logarithmic scale to detect the magnitude of the violation
    jitter_levels = [0.0, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    
    m = M.shape[0]
    I = torch.eye(m, device=M.device, dtype=M.dtype)
    
    for jitter in jitter_levels:
        if jitter > max_jitter:
            break
            
        # Construct candidate matrix
        if jitter > 0:
            M_curr = M + jitter * I
        else:
            M_curr = M
            
        try:
            # linalg.cholesky throws RuntimeError if not SPD
            L = torch.linalg.cholesky(M_curr)
            return L, jitter, True
        except RuntimeError:
            continue
            
    # If all attempts fail
    return None, max_jitter, False

def diagnose_core_matrix(W, G, sigma, delta):
    """
    Performs a deep numerical autopsy of the Core Kernel Matrix.
    
    M = sigma * W + sigma * delta * I + G
    
    This function computes expensive metrics (Eigenvalues) that we wouldn't 
    run in production, but are essential for the 'Analysis' phase of the paper.
    
    Args:
        W (Tensor): Core interaction matrix [m, m].
        G (Tensor): Gram matrix [m, m].
        sigma (float): Global damping.
        delta (float): Kernel regularization.
        
    Returns:
        metrics (dict): Contains 'min_eig', 'cond_num', 'jitter_needed', etc.
    """
    m = W.shape[0]
    device = W.device
    
    # 1. Reconstruct the Core Matrix exactly as Woodbury would
    regularization = (sigma * delta) * torch.eye(m, device=device)
    M = sigma * W + regularization + G
    
    metrics = {}
    
    # 2. Spectral Analysis (Eigenvalues)
    # Using eigvalsh since M is symmetric.
    # Note: On M1 MPS, eigvalsh is generally stable for m=256.
    try:
        eigs = torch.linalg.eigvalsh(M)
        
        # Determine the range
        min_eig = eigs[0].item()
        max_eig = eigs[-1].item()
        
        metrics['min_eig'] = min_eig
        metrics['max_eig'] = max_eig
        
        # Condition Number: abs(max) / abs(min)
        # Clamp denominator to avoid div-by-zero or explosion for logging
        denom = abs(min_eig) if abs(min_eig) > 1e-12 else 1e-12
        metrics['condition_num'] = abs(max_eig) / denom
        
        # Mathematical SPD check (ignoring numerical noise)
        metrics['is_spd_math'] = min_eig > 0.0
        
    except RuntimeError as e:
        logger.warning(f"Eigenvalue decomposition failed: {e}")
        metrics['min_eig'] = float('nan')
        metrics['condition_num'] = float('inf')
        metrics['is_spd_math'] = False

    # 3. Functional Analysis (Cholesky / Jitter)
    # This checks if the matrix is "computationally SPD"
    _, jitter_used, success = robust_cholesky_with_jitter(M)
    
    metrics['jitter_needed'] = jitter_used
    metrics['cholesky_success'] = success
    
    return metrics

def calc_gram_error(G_fp32, G_fp8):
    """
    Calculates the Relative Frobenius Norm Error between the ideal and quantized Gram matrices.
    
    metric = || G_fp32 - G_fp8 ||_F / || G_fp32 ||_F
    
    This is used to verify Assumption 6.3 and Appendix B in the paper.
    
    Args:
        G_fp32 (Tensor): The 'Ground Truth' Gram matrix computed with full precision.
        G_fp8 (Tensor): The Gram matrix computed from the Simulated FP8 sketch.
        
    Returns:
        float: Relative error.
    """
    # Ensure they are on same device/type for comparison
    target = G_fp32.to(G_fp8.device)
    
    diff_norm = torch.norm(target - G_fp8, p='fro')
    base_norm = torch.norm(target, p='fro')
    
    # Avoid div-by-zero if G is zero (init)
    if base_norm < 1e-9:
        return 0.0
        
    return (diff_norm / base_norm).item()