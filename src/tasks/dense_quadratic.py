import torch
import os
import logging

logger = logging.getLogger(__name__)

class DenseQuadraticTask:
    """
    Synthetic Dense Quadratic Optimization Task.
    Objective: f(x) = 1/2 * x^T H x + b^T x
    Gradient:  g(x) = H x + b
    Hessian:   H (Constant)
    
    This task is designed to be M1-friendly (Explicit Tensors) while providing
    mathematically rigorous test cases for Nystrom approximation.
    """

    def __init__(self, dim=2000, condition_number=1e6, spectrum='spiked', 
                 device='mps', dtype=torch.float32, data_path=None):
        """
        Args:
            dim (int): Dimension of the problem (d).
            condition_number (float): Condition number (lambda_max / lambda_min).
            spectrum (str): 'spiked' (Neural Net like) or 'power_law'.
            device (str): Compute device.
            data_path (str): Path to save/load the problem definition (H, b).
        """
        self.dim = dim
        self.cond = condition_number
        self.spectrum_type = spectrum
        self.device = device
        self.dtype = dtype
        
        # Data containers
        self.H = None
        self.b = None
        self.x_opt = None # Ground truth solution (if constructable)
        
        # Load or Generate
        if data_path and os.path.exists(data_path):
            self.load_data(data_path)
        else:
            self.generate_data()
            if data_path:
                self.save_data(data_path)
                
        # Move to device
        self.H = self.H.to(device=self.device, dtype=self.dtype)
        self.b = self.b.to(device=self.device, dtype=self.dtype)
        
    def generate_data(self):
        logger.info(f"Generating Dense Quadratic Data (d={self.dim}, cond={self.cond}, type={self.spectrum_type})...")
        
        # 1. Generate Eigenvalues
        # Log-spaced spectrum from 1/cond to 1.0
        min_eig = 1.0 / self.cond
        max_eig = 1.0
        
        eigs = torch.zeros(self.dim)
        
        if self.spectrum_type == 'spiked':
            # Top 5% eigenvalues capture 95% of energy
            spike_dim = max(1, int(0.05 * self.dim))
            
            # Spikes
            eigs[:spike_dim] = torch.linspace(0.8, max_eig, spike_dim)
            
            # Noise Floor (Long Tail)
            eigs[spike_dim:] = torch.linspace(min_eig, min_eig * 10, self.dim - spike_dim)
            
        else:
            # Power Law Decay (Standard Ill-conditioned)
            # lambda_i = i^(-alpha)
            idxs = torch.arange(1, self.dim + 1).float()
            # find alpha such that dim^(-alpha) = min_eig
            alpha = -math.log(min_eig) / math.log(self.dim)
            eigs = 1.0 / (idxs ** alpha)
            
        # 2. Generate Random Orthogonal Matrix Q
        # We perform QR decomposition on a random matrix to get Q
        # This ensures H = Q * diag(eigs) * Q^T is valid
        # On M1, for d=2000, this takes a few seconds.
        X = torch.randn(self.dim, self.dim)
        Q, _ = torch.linalg.qr(X)
        
        # 3. Construct H
        # H = Q @ diag(eigs) @ Q.T
        Lambda = torch.diag(eigs)
        self.H = Q @ Lambda @ Q.T
        
        # 4. Generate random solution x_opt and derive b
        # Let x_opt be random unit vector scaled
        self.x_opt = torch.randn(self.dim)
        # We want min f(x) at x_opt.
        # Grad at x_opt = H*x_opt + b = 0  =>  b = -H*x_opt
        self.b = -torch.mv(self.H, self.x_opt)
        
        logger.info("Data generation complete.")

    def save_data(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({'H': self.H, 'b': self.b, 'x_opt': self.x_opt}, path)
        logger.info(f"Data saved to {path}")

    def load_data(self, path):
        logger.info(f"Loading data from {path}...")
        data = torch.load(path, map_location='cpu')
        self.H = data['H']
        self.b = data['b']
        self.x_opt = data.get('x_opt', None)
        self.dim = self.H.shape[0]

    def closure(self, x):
        """
        Computes Loss and Gradient.
        
        Args:
            x (Tensor): Current parameters [d].
            
        Returns:
            loss (float): Scalar loss.
            grad (Tensor): Gradient vector [d].
        """
        # H is symmetric
        # f(x) = 0.5 * x^T H x + b^T x
        
        # Hx
        Hx = torch.mv(self.H, x)
        
        # Loss
        # 0.5 * x dot Hx + b dot x
        quad_term = 0.5 * torch.dot(x, Hx)
        lin_term = torch.dot(self.b, x)
        loss = quad_term + lin_term
        
        # Gradient
        # g = Hx + b
        grad = Hx + self.b
        
        return loss, grad

    def hvp(self, v):
        """
        Explicit Hessian-Vector Product Oracle.
        
        Args:
            v (Tensor): Vector or Matrix [d] or [d, batch].
            
        Returns:
            Hv (Tensor): Result of H * v.
        """
        return torch.mm(self.H, v) if v.dim() > 1 else torch.mv(self.H, v)

# CLI entry point to pre-generate data
if __name__ == "__main__":
    import argparse
    import math
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=2000)
    parser.add_argument("--cond", type=float, default=1e6)
    parser.add_argument("--spectrum", type=str, default="spiked", choices=["spiked", "power_law"])
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    logging.basicConfig(level=logging.INFO)
    
    task = DenseQuadraticTask(
        dim=args.dim, 
        condition_number=args.cond, 
        spectrum=args.spectrum, 
        device='cpu' # Generate on CPU to be safe, then run on MPS
    )
    task.save_data(args.out)