import math

class CommsModel:
    """
    Hardware Emulator for Distributed Communication (InfiniBand/NVLink).
    
    Paper Reference: Section 5.2 (Communication Complexity)
    
    Purpose:
    Simulates the wall-clock time cost of collective operations (All-Reduce)
    on a logical cluster, running physically on a single M1 process.
    
    The cost model follows the standard LogP/Alpha-Beta model for Ring All-Reduce:
        Time = Alpha (Latency) + Beta (Bandwidth inverse) * Effective_Volume
        
    Correction from previous versions:
    - Bandwidth inputs are typically in Gbps (Gigabits/sec).
    - Payload sizes are in Bytes.
    - We strictly convert Gbps -> GB/s to ensure Beta has the correct unit (sec/byte).
    """

    def __init__(self, num_shards=32, bandwidth_gbps=400.0, latency_us=50.0):
        """
        Args:
            num_shards (int): Number of GPUs/Nodes (P). Default: 32 (Paper setup).
            bandwidth_gbps (float): Network bandwidth in Gigabits per second.
                                    Default: 400.0 (NVIDIA Quantum-2 InfiniBand).
            latency_us (float): Base interconnect latency in microseconds.
                                Default: 50.0 us.
        """
        self.P = num_shards
        
        # [CRITICAL FIX]: Convert Gbps (bits) to Bytes/sec
        # 1 Gbps = 1e9 bits/s
        # 1 Byte = 8 bits
        bytes_per_sec = (bandwidth_gbps * 1e9) / 8.0
        
        # Beta: Time to transmit 1 byte (seconds/byte)
        self.beta = 1.0 / bytes_per_sec
        
        # Alpha: Latency (seconds)
        self.alpha = latency_us * 1e-6
        
        # Statistics accumulator
        self.stats = {
            'total_time': 0.0,
            'total_payload_bytes': 0,
            'total_effective_bytes': 0,
            'ops_count': 0
        }

    def log_all_reduce(self, payload_bytes, name="unknown"):
        """
        Simulates a Blocking Ring All-Reduce operation.
        
        Mathematical Model:
        For a payload of size M bytes across P nodes, Ring All-Reduce 
        (Scatter-Reduce + All-Gather) transmits:
            Traffic = 2 * (P - 1) / P * M
        
        Args:
            payload_bytes (int): Size of the tensor being reduced (in Bytes).
                                 e.g., m*m*4 for a float32 Gram matrix.
            name (str): Label for the operation (e.g., "Gram", "Gradient").
            
        Returns:
            dict: {
                'time_cost': float (seconds),
                'effective_bytes': int,
                'payload_bytes': int
            }
        """
        if self.P <= 1:
            # Single device: No communication cost
            return {
                'time_cost': 0.0,
                'effective_bytes': 0,
                'payload_bytes': payload_bytes
            }

        # 1. Calculate Ring Factor
        # As P -> infinity, this approaches 2.0 (sending data twice).
        ring_factor = 2.0 * (self.P - 1) / self.P
        
        # 2. Calculate Effective Traffic
        # This is the actual amount of data pushed through the wire per node.
        effective_bytes = int(payload_bytes * ring_factor)
        
        # 3. Calculate Time
        # Linear Latency-Bandwidth Model
        transmission_time = effective_bytes * self.beta
        total_time = self.alpha + transmission_time
        
        # 4. Update Stats
        self.stats['total_time'] += total_time
        self.stats['total_payload_bytes'] += payload_bytes
        self.stats['total_effective_bytes'] += effective_bytes
        self.stats['ops_count'] += 1
        
        return {
            'time_cost': total_time,
            'effective_bytes': effective_bytes,
            'payload_bytes': payload_bytes,
            'name': name
        }

    def reset_stats(self):
        """Clears cumulative statistics."""
        for k in self.stats:
            self.stats[k] = 0

    def __repr__(self):
        return (f"<CommsModel P={self.P}, BW={1.0/self.beta/1e9*8:.1f}Gbps, "
                f"Lat={self.alpha*1e6:.1f}us>")