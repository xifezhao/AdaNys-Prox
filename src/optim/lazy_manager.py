import torch
import logging

logger = logging.getLogger(__name__)

class LazyCacheManager:
    """
    Manages the lifecycle of the Nyström preconditioner cache.
    Implements the 'Dynamic Force Update' mechanism described in Section 5.3.
    
    Paper Reference:
    - Eq. (18): Staleness Limit (Max Age)
    - Eq. (19): Optimization Failure (Line Search)
    - Eq. (20): Geometry Shift (Gradient Orthogonality)
    """

    def __init__(self, config=None, device='mps'):
        """
        Args:
            config (dict): Configuration dictionary for triggers.
                - tau_max (int): Maximum reuse steps (default: 50).
                - use_triggers (bool): Master switch.
                - trigger_thresholds (dict):
                    - orthogonality (float): Cosine threshold (e.g., 0.05).
                    - line_search_failures (int): Max consecutive failures.
            device (str): Device for tensor operations.
        """
        self.device = device
        self.config = config or {}
        
        # Default Settings (aligned with Platinum Spec)
        self.tau_max = self.config.get('tau_max', 50)
        self.use_triggers = self.config.get('use_triggers', True)
        
        thresholds = self.config.get('trigger_thresholds', {})
        self.ortho_threshold = thresholds.get('orthogonality', 0.05)
        self.fail_threshold = thresholds.get('line_search_failures', 2)
        
        # State tracking
        self.last_update_step = -1
        self.consecutive_failures = 0
        
        logger.info(f"LazyManager Initialized: MaxAge={self.tau_max}, "
                    f"Triggers={'ON' if self.use_triggers else 'OFF'}")

    def check_triggers(self, step, current_grad, prev_grad=None, line_search_failed=False):
        """
        Evaluates stability conditions to determine if a Force Update is required.
        
        Args:
            step (int): Current global step.
            current_grad (Tensor): Flattened gradient vector at current step.
            prev_grad (Tensor): Flattened gradient from previous step.
            line_search_failed (bool): Whether the optimizer struggled to find descent.
            
        Returns:
            should_update (bool): True if update is required.
            reason (str): 'init', 'max_age', 'orthogonality', 'line_search', or 'none'.
        """
        # 0. Initialization Trigger
        if self.last_update_step < 0:
            return True, "init"

        # 1. Staleness Limit (Eq. 18)
        # This is a hard constraint, applies even if triggers are 'disabled' in some contexts,
        # though usually we consider this part of the trigger system.
        age = step - self.last_update_step
        if age >= self.tau_max:
            return True, "max_age"

        # If safety triggers are explicitly disabled (e.g., Experiment A2), stop here.
        if not self.use_triggers:
            return False, "none"

        # 2. Optimization Failure (Eq. 19)
        if line_search_failed:
            self.consecutive_failures += 1
        else:
            self.consecutive_failures = 0
            
        if self.consecutive_failures >= self.fail_threshold:
            return True, "line_search_fail"

        # 3. Geometry Shift / Gradient Orthogonality (Eq. 20)
        # Detects if the landscape takes a sharp turn (e.g., traversing a ridge).
        if prev_grad is not None:
            # Efficient cosine similarity on flattened vectors
            # cos = (g_t . g_{t-1}) / (|g_t| * |g_{t-1}|)
            # Add epsilon to prevent div-by-zero
            dot = torch.dot(current_grad, prev_grad)
            norm_curr = torch.norm(current_grad)
            norm_prev = torch.norm(prev_grad)
            
            cosine = dot / (norm_curr * norm_prev + 1e-8)
            
            if cosine < self.ortho_threshold:
                # Value < 0.05 implies nearly orthogonal or opposing directions
                return True, "orthogonality"

        return False, "none"

    def update_state(self, step):
        """
        Called by the optimizer AFTER a successful Nyström update.
        Resets counters and updates timestamps.
        """
        self.last_update_step = step
        self.consecutive_failures = 0