import torch

class FP8E4M3Simulator:
    """
    Hardware Emulator for FP8 (E4M3) format characteristics.
    
    Paper Reference: Section 5.1 & Appendix B
    
    Characteristics simulated:
    1. Precision: ~3 bit mantissa. This is modeled not as random noise, 
       but as discrete rounding to the nearest representable 'lattice' point 
       defined by the relative magnitude.
    2. Dynamic Range: [-448, 448]. Values outside this are hard-clamped (saturated).
    3. Subnormal Flush: Values smaller than 2^-9 are flushed to zero (absolute threshold).
    4. Storage Model: 1 byte per element + 4 bytes per column (scaler).
    
    Note: This runs on M1 MPS/CPU using FP32 math to logically verify 
    the numerical behavior of FP8, without requiring actual FP8 hardware instructions.
    """

    def __init__(self, device='mps'):
        self.device = device
        
        # --- E4M3 Specifications ---
        # Exponent bias: 7
        # Max normal: 1.111 * 2^8 = 448
        # Min normal: 1.000 * 2^-6 = 0.015625
        # Min subnormal: 0.001 * 2^-6 = 2^-9 approx 0.00195
        
        self.mantissa_bits = 3
        self.max_val = 448.0
        self.subnormal_threshold = 2**-9  # Absolute threshold for flush-to-zero
        
        # Precision floor to avoid division by zero during rounding step calculation
        self.epsilon = 1e-7 

    def quantize_store(self, tensor):
        """
        Simulates the storage pipeline: 
        FP32 Input -> Col-Scale -> FP8 Rounding -> Storage -> Dequantize to FP32 (for calc).
        
        Args:
            tensor (torch.Tensor): Input sketch matrix C_t [d, m] in FP32.
            
        Returns:
            reconstructed (torch.Tensor): The effective values used for computation, 
                                          containing quantization artifacts.
            storage_bytes (int): Theoretical memory usage in bytes.
        """
        # ---------------------------------------------------------
        # 1. Column-wise Scaling (Paper Eq. 16)
        # ---------------------------------------------------------
        # We map the dynamic range of each column to the "working range" of FP8.
        # s_j = max(|C_:,j|)
        scales = tensor.abs().max(dim=0).values.clamp(min=1e-6)
        
        # Normalize to conceptually [-1, 1] (relative to the scale)
        # Note: We keep high precision here before rounding
        norm_tensor = tensor / scales 

        # ---------------------------------------------------------
        # 2. Lattice Rounding (Simulating Mantissa Limits)
        # ---------------------------------------------------------
        # In floating point, precision is relative. The grid spacing depends on magnitude.
        # Step size delta(x) ~= |x| * 2^(-mantissa_bits)
        
        abs_norm = norm_tensor.abs()
        
        # Calculate the local grid step size for each element
        relative_step = abs_norm * (2 ** -self.mantissa_bits)
        
        # Ensure step size doesn't vanish (simulation stability)
        step_size = torch.maximum(
            torch.tensor(self.epsilon, device=self.device), 
            relative_step
        )
        
        # Discretize: Round to nearest multiple of local step size
        # x_quant = round(x / step) * step
        # This creates the "staircase" effect of low-bit floating point
        quantized_norm = torch.round(norm_tensor / step_size) * step_size

        # ---------------------------------------------------------
        # 3. Reconstruction (Dequantization)
        # ---------------------------------------------------------
        # Bring back to original magnitude
        reconstructed = quantized_norm * scales

        # ---------------------------------------------------------
        # 4. Range Saturation (Simulating Exponent Limits)
        # ---------------------------------------------------------
        # E4M3 cannot represent values > 448. Even with scaling, if the 
        # distribution is extremely heavy-tailed, outliers might clip.
        # Note: In standard scaling, max(abs) maps to 1.0, so this usually 
        # doesn't trigger unless the scale itself overflows (rare).
        # We keep it for rigor.
        reconstructed = reconstructed.clamp(-self.max_val, self.max_val)

        # ---------------------------------------------------------
        # 5. Flush Subnormals (Absolute Threshold)
        # ---------------------------------------------------------
        # Values that are too small in absolute terms cannot be represented.
        # This is critical for sparse data or decaying gradients.
        mask_flush = reconstructed.abs() < self.subnormal_threshold
        
        # We perform an in-place modification to simulate the "flush"
        # Using clone to ensure gradient safety if this were diff-able (it isn't)
        reconstructed = reconstructed.clone()
        reconstructed[mask_flush] = 0.0

        # ---------------------------------------------------------
        # 6. Theoretical Memory Accounting
        # ---------------------------------------------------------
        d, m = tensor.shape
        # Payload: 1 byte per element (FP8)
        # Metadata: 4 bytes per column (FP32 scaler)
        storage_bytes = (d * m * 1) + (m * 4)
        
        return reconstructed, storage_bytes

    def to(self, device):
        """Move internal constants to device."""
        self.device = device
        return self