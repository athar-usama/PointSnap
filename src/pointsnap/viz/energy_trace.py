"""Renders an MRF energy trace as an ASCII bar-per-iteration table for the
README -- a direct visual proof that checkerboard ICM never increases the
energy it optimizes, without resorting to a plotting-library line chart."""

from __future__ import annotations


def format_energy_trace(energy_trace: list[float], width: int = 40) -> str:
    lo, hi = min(energy_trace), max(energy_trace)
    span = max(hi - lo, 1e-9)
    lines = ["iter  energy        descent"]
    for i, e in enumerate(energy_trace):
        frac = (e - lo) / span
        filled = round(frac * width)
        bar = "#" * filled + "." * (width - filled)
        label = "init" if i == 0 else f"{i:>4d}"
        lines.append(f"{label}  {e:>10.2f}   [{bar}]")
    return "\n".join(lines)
