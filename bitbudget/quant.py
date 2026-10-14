"""Low-level quantization helpers.

Everything here is a thin layer over ``mx.quantize`` / ``mx.dequantize`` so that
the error we *optimize* during calibration is exactly the error of the kernel
that runs at inference time (MLX affine, group-wise, asymmetric quantization).
"""

from __future__ import annotations

import mlx.core as mx

# Bit-widths supported by MLX's affine quantization kernels.
SUPPORTED_BITS = (2, 3, 4, 5, 6, 8)


def qdq(w: mx.array, bits: int, group_size: int) -> mx.array:
    """Quantize then dequantize ``w`` with MLX affine group quantization.

    Returns an array of the same shape and dtype as ``w`` holding the values the
    quantized kernel will actually see.
    """
    if bits not in SUPPORTED_BITS:
        raise ValueError(f"bits={bits} not supported; choose from {SUPPORTED_BITS}")
    wq, scales, biases = mx.quantize(w, group_size=group_size, bits=bits)
    return mx.dequantize(wq, scales, biases, group_size=group_size, bits=bits).astype(
        w.dtype
    )


def group_overhead_bits(dtype: mx.Dtype) -> int:
    """Bits of metadata stored per quantization group.

    MLX affine quantization stores one scale and one bias per group, both in
    the dtype of the original weight (16-bit for bf16/fp16 checkpoints).
    """
    return 2 * dtype.size * 8


def effective_bpw(bits: int, group_size: int, dtype: mx.Dtype = mx.bfloat16) -> float:
    """Effective bits per weight including per-group scale/bias metadata.

    Example: 4-bit, group size 64, bf16 metadata -> 4 + 32/64 = 4.5 bpw.
    """
    return bits + group_overhead_bits(dtype) / group_size


def is_quantizable(weight: mx.array, group_size: int) -> bool:
    return weight.ndim == 2 and weight.shape[-1] % group_size == 0
