"""Energy per token from macOS ``powermetrics`` logs.

``scripts/measure_energy.sh`` records ``sudo powermetrics`` while
``bitbudget bench`` runs and saves the benchmark's start/end timestamps.
This module averages the package power samples that fall inside that window.

Caveat: package power includes the OS and anything else running; measure an
idle baseline with the same script (``--idle``) and subtract it for a cleaner
estimate of the model's incremental energy.
"""

from __future__ import annotations

import re
from datetime import datetime

_HEADER = re.compile(r"\*\*\* Sampled system activity \((.+?)\) \(([\d.]+)ms elapsed\)")
_COMBINED = re.compile(r"Combined Power \(CPU \+ GPU \+ ANE\):\s*([\d.]+)\s*mW")
_GPU = re.compile(r"^GPU Power:\s*([\d.]+)\s*mW", re.M)


def _parse_time(s: str) -> float | None:
    for fmt in ("%a %b %d %H:%M:%S %Y %z", "%a %b %d %H:%M:%S %Y"):
        try:
            return datetime.strptime(s.strip(), fmt).timestamp()
        except ValueError:
            continue
    return None


def parse_powermetrics(text: str) -> list[dict]:
    """Return one dict per sample: ``{"t": epoch_or_None, "dt_ms", "combined_mw",
    "gpu_mw"}``."""
    samples = []
    blocks = re.split(r"(?=\*\*\* Sampled system activity)", text)
    for blk in blocks:
        h = _HEADER.search(blk)
        c = _COMBINED.search(blk)
        if not h or not c:
            continue
        g = _GPU.search(blk)
        samples.append(
            {
                "t": _parse_time(h.group(1)),
                "dt_ms": float(h.group(2)),
                "combined_mw": float(c.group(1)),
                "gpu_mw": float(g.group(1)) if g else None,
            }
        )
    return samples


def energy_per_token(
    samples: list[dict],
    t_start: float,
    t_end: float,
    tokens: int,
    idle_mw: float = 0.0,
) -> dict:
    """Joules per generated token over ``[t_start, t_end]``.

    Falls back to all samples when the log has no parseable timestamps.
    """
    inside = [s for s in samples if s["t"] is not None and t_start <= s["t"] <= t_end]
    used = inside or samples
    if not used:
        raise ValueError("no powermetrics samples found")
    mean_mw = sum(s["combined_mw"] for s in used) / len(used)
    net_mw = max(mean_mw - idle_mw, 0.0)
    seconds = t_end - t_start
    joules = net_mw / 1000.0 * seconds
    return {
        "mean_power_w": mean_mw / 1000.0,
        "net_power_w": net_mw / 1000.0,
        "seconds": seconds,
        "joules": joules,
        "j_per_token": joules / max(tokens, 1),
        "samples": len(used),
        "windowed": bool(inside),
    }
