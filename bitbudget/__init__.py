"""BitBudget: salience-guided mixed-precision quantization for on-device LLMs (MLX)."""

from .allocate import solve_dp, solve_lagrangian
from .pipeline import (
    Config,
    Curves,
    allocate,
    apply_plan,
    compute_curves,
    quantize_in_memory,
    quantize_with_plan,
    uniform_plan,
)

__all__ = [
    "Config",
    "Curves",
    "allocate",
    "apply_plan",
    "compute_curves",
    "quantize_in_memory",
    "quantize_with_plan",
    "solve_dp",
    "solve_lagrangian",
    "uniform_plan",
]
__version__ = "0.1.0"
