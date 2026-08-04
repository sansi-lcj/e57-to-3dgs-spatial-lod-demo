#!/usr/bin/env python3
"""Remove obvious floaters from an OpenSplat Gaussian PLY."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from e57gs.gaussian_ply import filter_gaussian_ply


def triplet(value: str) -> tuple[float, float, float]:
    parts = tuple(float(part) for part in value.split(","))
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected X,Y,Z")
    return parts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--min-opacity", type=float, default=0.03)
    parser.add_argument("--scale-mode", choices=("max", "needle"), default="needle")
    parser.add_argument("--max-scale-m", type=float, default=0.10)
    parser.add_argument("--second-scale-m", type=float, default=0.05)
    parser.add_argument("--min-aspect-ratio", type=float, default=4.0)
    parser.add_argument("--bounds-min", type=triplet, required=True)
    parser.add_argument("--bounds-max", type=triplet, required=True)
    args = parser.parse_args()

    result = filter_gaussian_ply(
        args.input.resolve(),
        args.output.resolve(),
        min_opacity=args.min_opacity,
        max_scale_m=args.max_scale_m,
        bounds_min_m=args.bounds_min,
        bounds_max_m=args.bounds_max,
        scale_mode=args.scale_mode,
        second_scale_m=args.second_scale_m,
        min_aspect_ratio=args.min_aspect_ratio,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
