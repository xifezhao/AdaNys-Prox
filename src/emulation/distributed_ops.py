import torch
from .comms_model import CommsModel

class LogicalCluster:
    """
    Simulates a Distributed Data Parallel (DDP) environment on a single machine.
    
    Paper Reference: Section 5.2 (Distributed Lazy Updates)
    
    Responsibilities:
    1. Logical Sharding: Splits global parameters/sketches into P chunks.
    2. Local Computation: Executes math kernels on each chunk independently.
    3. Global Aggregation: Simulates All-Reduce via CommsModel accounting.
    
    This ensures that our experiment validates the *feasibility* of the distributed 
    algorithm, proving that operations like Gram matrix construction strictly 
    adhere to the "Local Compute -> Small Communication" paradigm.
    """

    def __init__(self, num_shards=32, comms_model: CommsModel = None):
        """
        Args:
            num_shards (int): Number of simulated nodes (P).
            comms_model (CommsModel): Instance for logging communication overhead.
        """
        self.P = num_shards
        self.comms = comms_model
        
    def shard_tensor(self, tensor, dim=0):
        """
        Splits a global tensor into P logical shards along the specified dimension.
        Usually splits parameters/gradients along dim 0 (Data Parallelism).
        """
        if tensor is None:
            return [None] * self.P
        
        # Use tensor_split to handle cases where size isn't divisible by P
        return torch.tensor_split(tensor, self.P, dim=dim)

    def compute_distributed_gram(self, sharded_C):
        """
        Simulates the construction of the global Gram matrix G_t = C^T * C.
        
        Complexity:
            Compute: O(m^2 * d/P) per node [Parallel]
            Comm:    O(m^2) All-Reduce [Bottleneck]
            
        Args:
            sharded_C (list[Tensor]): List of P tensors, each [d_i, m].
            
        Returns:
            G_global (Tensor): [m, m] matrix.
        """
        # 1. Local Computation Phase (Simulated Parallelism)
        local_grams = []
        for C_i in sharded_C:
            # G_i = C_i^T @ C_i
            # Note: We compute in FP32 as mandated by Paper Section 5.1
            local_grams.append(torch.mm(C_i.t(), C_i))
            
        # 2. Communication Phase (All-Reduce)
        # We sum the local results to get the global Gram
        G_global = sum(local_grams)
        
        # 3. Log Communication Cost
        if self.comms:
            # Payload: m * m * 4 bytes (FP32)
            m = G_global.size(0)
            payload_bytes = m * m * 4
            self.comms.log_all_reduce(payload_bytes, name="Gram_Matrix_Update")
            
        return G_global

    def compute_distributed_projection(self, sharded_C, sharded_v):
        """
        Simulates the projection u = C^T * v.
        Used in the ANP-Newton reuse phase.
        
        Complexity:
            Compute: O(m * d/P) per node
            Comm:    O(m) All-Reduce
            
        Args:
            sharded_C (list[Tensor]): List of P tensors [d_i, m].
            sharded_v (list[Tensor]): List of P tensors [d_i].
            
        Returns:
            u_global (Tensor): [m] vector.
        """
        # 1. Local Computation
        local_projections = []
        for C_i, v_i in zip(sharded_C, sharded_v):
            # u_i = C_i^T @ v_i
            # Reshape v_i to [d_i, 1] for matmul, then flatten
            u_i = torch.mm(C_i.t(), v_i.unsqueeze(1)).squeeze(1)
            local_projections.append(u_i)
            
        # 2. Communication (All-Reduce Sum)
        u_global = sum(local_projections)
        
        # 3. Log Cost
        if self.comms:
            # Payload: m * 4 bytes
            m = u_global.size(0)
            payload_bytes = m * 4
            self.comms.log_all_reduce(payload_bytes, name="Reuse_Projection")
            
        return u_global

    def compute_distributed_diagonal(self, sharded_C, K_global):
        """
        Computes diagonal elements for ANP-Diagonal protocol.
        
        Logic: h_i = diag(C_i * K * C_i^T).
        Crucially, this requires NO communication because K is globally broadcasted 
        (negligible cost) and each row depends only on local C data.
        
        Args:
            sharded_C (list[Tensor]): P shards of sketch.
            K_global (Tensor): The small [m, m] kernel inverse.
            
        Returns:
            h_sharded (list[Tensor]): List of P diagonal segments.
        """
        h_sharded = []
        
        # 1. Local Compute Only (Embarrassingly Parallel)
        for C_i in sharded_C:
            # We want diag(C_i @ K @ C_i.T) efficiently
            # Let M = C_i @ K [d_i, m]
            M_i = torch.mm(C_i, K_global)
            
            # The diagonal of (M_i @ C_i.T) is the sum of element-wise product
            # diag_vals = sum(M_i * C_i, dim=1)
            diag_i = (M_i * C_i).sum(dim=1)
            
            h_sharded.append(diag_i)
            
        # No All-Reduce needed for h itself!
        # This confirms the efficiency of ANP-Diagonal.
        
        return h_sharded

    def gather_tensor(self, sharded_tensor):
        """
        Helper to reconstruct full tensor for metrics calculation (e.g. Loss).
        Not part of the algorithm's runtime loop.
        """
        return torch.cat(sharded_tensor, dim=0)