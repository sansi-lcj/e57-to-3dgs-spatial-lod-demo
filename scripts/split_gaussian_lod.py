#!/usr/bin/env python3
"""Split one Gaussian PLY into deterministic additive spatial LoD PLYs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from e57gs.lod_tiles import build_additive_lod_plys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--base-voxel-m", type=float, default=0.06)
    parser.add_argument("--mid-voxel-m", type=float, default=0.02)
    parser.add_argument("--tile-size-m", type=float, default=2.0)
    parser.add_argument("--mid-load-distance-m", type=float, default=4.5)
    parser.add_argument("--mid-unload-distance-m", type=float, default=5.5)
    parser.add_argument("--fine-load-distance-m", type=float, default=2.75)
    parser.add_argument("--fine-unload-distance-m", type=float, default=3.6)
    parser.add_argument("--base-tangent-scale-m", type=float, default=0.0)
    parser.add_argument("--mid-tangent-scale-m", type=float, default=0.0)
    parser.add_argument("--mid-view-distance-m", type=float, default=14.0)
    parser.add_argument("--mid-view-half-angle-degrees", type=float, default=85.0)
    parser.add_argument("--fine-view-distance-m", type=float, default=14.0)
    parser.add_argument("--fine-view-half-angle-degrees", type=float, default=65.0)
    args = parser.parse_args()
    result = build_additive_lod_plys(
        args.input.resolve(),
        args.output.resolve(),
        base_voxel_m=args.base_voxel_m,
        mid_voxel_m=args.mid_voxel_m,
        tile_size_m=args.tile_size_m,
        mid_load_distance_m=args.mid_load_distance_m,
        mid_unload_distance_m=args.mid_unload_distance_m,
        fine_load_distance_m=args.fine_load_distance_m,
        fine_unload_distance_m=args.fine_unload_distance_m,
        base_tangent_scale_m=args.base_tangent_scale_m or None,
        mid_tangent_scale_m=args.mid_tangent_scale_m or None,
        mid_view_distance_m=args.mid_view_distance_m,
        mid_view_half_angle_degrees=args.mid_view_half_angle_degrees,
        fine_view_distance_m=args.fine_view_distance_m,
        fine_view_half_angle_degrees=args.fine_view_half_angle_degrees,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
