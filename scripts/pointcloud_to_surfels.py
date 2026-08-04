#!/usr/bin/env python3
"""Generate a geometry-faithful SH0 Gaussian PLY from XYZ/RGB/normal PLY."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from e57gs.surfels import point_cloud_to_gaussian_surfels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--density-voxel", type=float, default=0.02)
    parser.add_argument("--tangent-factor", type=float, default=0.55)
    parser.add_argument("--min-scale", type=float, default=0.002)
    parser.add_argument("--max-scale", type=float, default=0.012)
    parser.add_argument("--normal-ratio", type=float, default=0.12)
    parser.add_argument("--opacity", type=float, default=0.85)
    parser.add_argument("--sh-degree", type=int, choices=range(4), default=0)
    parser.add_argument("--opensplat-iteration", type=int)
    args = parser.parse_args()
    result = point_cloud_to_gaussian_surfels(
        args.input,
        args.output,
        density_voxel_m=args.density_voxel,
        tangent_factor=args.tangent_factor,
        min_tangent_scale_m=args.min_scale,
        max_tangent_scale_m=args.max_scale,
        normal_scale_ratio=args.normal_ratio,
        opacity=args.opacity,
        sh_degree=args.sh_degree,
        opensplat_iteration=args.opensplat_iteration,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
