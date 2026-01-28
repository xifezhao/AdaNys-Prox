import torch
import os
import logging
import math

logger = logging.getLogger(__name__)

class SparseLogisticTask:
    """
    High-Dimensional Sparse Logistic Regression Task.
    
    Objective (Smooth part): 
        f(w) = -1/N * sum( y_i * log(p_i) + (1-y_i) * log(1-p_i) )
    
    Gradient:
        g(w) = 1/N * X^T (p - y)
        
    Hessian (Explicit Operator):
        H(w) = 1/N * X^T D X,  where D = diag(p * (1-p))
    
    This task is designed to verify the "Metric Consistency" hypothesis.
    The optimizer is responsible for adding the L1 regularization term g(w) = lambda ||w||_1.
    """

    def __init__(self, dim=50000, num_samples=2000, sparsity=0.01, 
                 device='mps', dtype=torch.float32, data_path=None):
        """
        Args:
            dim (int): Feature dimension (d). Default M1 scale: 50k.
            num_samples (int): Number of data points (N).
            sparsity (float): Proportion of non-zero elements in ground truth (e.g., 0.01).
            device (str): Compute device.
            data_path (str): Path to save/load dataset.
        """
        self.dim = dim
        self.n_samples = num_samples
        self.sparsity = sparsity
        self.device = device
        self.dtype = dtype
        
        # Data containers
        self.X = None # Features [N, d]
        self.y = None # Labels [N]
        self.w_true = None # Ground truth weights (Sparse)
        
        # Current state cache for HVP (to avoid recomputing p)
        self.last_p = None
        
        # Load or Generate
        if data_path and os.path.exists(data_path):
            self.load_data(data_path)
        else:
            self.generate_data()
            if data_path:
                self.save_data(data_path)
                
        # Move to device
        self.X = self.X.to(device=self.device, dtype=self.dtype)
        self.y = self.y.to(device=self.device, dtype=self.dtype)
        self.w_true = self.w_true.to(device=self.device, dtype=self.dtype)
        
        # Pre-calculate Ground Truth Support for Metrics
        self.true_support = (self.w_true.abs() > 1e-6)

    def generate_data(self):
        logger.info(f"Generating Sparse LR Data (d={self.dim}, N={self.n_samples}, sparsity={self.sparsity})...")
        
        # 1. Generate Sparse Ground Truth Weights
        # k = number of active features
        k = int(self.dim * self.sparsity)
        self.w_true = torch.zeros(self.dim)
        
        # Random indices for support
        indices = torch.randperm(self.dim)[:k]
        # Random weights (avoid 0)
        values = torch.randn(k)
        self.w_true[indices] = values
        
        # 2. Generate Features X
        # Standard Gaussian features. 
        # Note: In real Criteo, features are sparse categorical. 
        # Here we use dense X to stress-test the optimizer's ability to 
        # find the sparse signal amidst dense noise.
        self.X = torch.randn(self.n_samples, self.dim)
        
        # Optional: Correlate features to make H ill-conditioned
        # (Skip for basic M1 demo to save generation time, but critical for rigorous testing)
        
        # 3. Generate Labels y
        # logits = Xw
        logits = torch.mv(self.X, self.w_true)
        probs = torch.sigmoid(logits)
        
        # y ~ Bernoulli(p)
        self.y = torch.bernoulli(probs)
        
        logger.info(f"Data generated. True Non-Zeros: {k}")

    def save_data(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({'X': self.X, 'y': self.y, 'w_true': self.w_true}, path)
        logger.info(f"Data saved to {path}")

    def load_data(self, path):
        logger.info(f"Loading data from {path}...")
        data = torch.load(path, map_location='cpu')
        self.X = data['X']
        self.y = data['y']
        self.w_true = data['w_true']
        self.dim = self.X.shape[1]
        self.n_samples = self.X.shape[0]

    def closure(self, w):
        """
        Computes Logistic Loss and Gradient.
        
        Args:
            w (Tensor): Current weights [d].
            
        Returns:
            loss (float): Scalar loss (BCE).
            grad (Tensor): Gradient vector [d].
        """
        # 1. Forward
        logits = torch.mv(self.X, w)
        
        # Numeric stability: use BCEWithLogits semantics
        # Loss = max(logits, 0) - logits * y + log(1 + exp(-abs(logits)))
        # PyTorch has a built-in for this:
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, self.y, reduction='mean'
        )
        
        # 2. Backward (Gradient)
        # g = 1/N * X^T (p - y)
        probs = torch.sigmoid(logits)
        error = probs - self.y
        grad = torch.mv(self.X.t(), error) / self.n_samples
        
        # Cache probs for HVP
        self.last_p = probs.detach()
        
        return loss, grad

    def hvp(self, v):
        """
        Explicit Hessian-Vector Product Oracle.
        
        Calculates H * v = (1/N) * X^T * [ p(1-p) * (X * v) ]
        
        Complexity: O(N * d), linear in dimensions, very efficient.
        Avoids O(d^2) Hessian matrix.
        
        Args:
            v (Tensor): Vector [d] or Matrix [d, m].
            
        Returns:
            Hv (Tensor): Result of H * v.
        """
        if self.last_p is None:
            # Should normally happen after closure, but for safety:
            raise RuntimeError("Must call closure() before hvp() to set state.")
            
        # 1. D = p * (1 - p) -> Diagonal of Hessian (N elements)
        # D shape: [N]
        D = self.last_p * (1.0 - self.last_p)
        
        # Handle batch v (d x m) or vector v (d)
        if v.dim() > 1:
            # Matrix case (Nystrom Sketching S_t)
            # Xv shape: [N, d] @ [d, m] -> [N, m]
            Xv = torch.mm(self.X, v)
            
            # Multiply by D (broadcasting across m columns)
            # D: [N, 1], Xv: [N, m]
            DXv = D.unsqueeze(1) * Xv
            
            # X^T (DXv)
            # [d, N] @ [N, m] -> [d, m]
            Hv = torch.mm(self.X.t(), DXv)
            
        else:
            # Vector case
            # Xv shape: [N]
            Xv = torch.mv(self.X, v)
            
            # D * Xv (Element-wise)
            DXv = D * Xv
            
            # X^T (DXv)
            Hv = torch.mv(self.X.t(), DXv)
            
        # Scale by 1/N
        return Hv / self.n_samples

    def get_support_metrics(self, w_current, threshold=1e-5):
        """
        Calculates sparsity metrics against Ground Truth.
        Used for "Platinum" evidence in Metric Consistency experiment.
        
        Args:
            w_current (Tensor): Current weights.
            threshold (float): Cutoff for considering a weight non-zero.
            
        Returns:
            metrics (dict): {precision, recall, f1, num_features}
        """
        with torch.no_grad():
            pred_support = (w_current.abs() > threshold)
            
            # Intersection (True Positives)
            tp = (pred_support & self.true_support).sum().item()
            
            # Selected (TP + FP)
            selected = pred_support.sum().item()
            
            # True (TP + FN)
            relevant = self.true_support.sum().item()
            
            precision = tp / selected if selected > 0 else 0.0
            recall = tp / relevant if relevant > 0 else 0.0
            
            f1 = 0.0
            if (precision + recall) > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
                
            return {
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'non_zeros': selected
            }

# CLI entry point to pre-generate data
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=50000)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--sparsity", type=float, default=0.01)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    logging.basicConfig(level=logging.INFO)
    
    # Generate on CPU to ensure memory safety during gen
    task = SparseLogisticTask(
        dim=args.dim, 
        num_samples=args.samples,
        sparsity=args.sparsity, 
        device='cpu' 
    )
    task.save_data(args.out)