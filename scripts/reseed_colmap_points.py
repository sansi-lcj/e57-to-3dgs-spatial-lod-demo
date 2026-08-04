#!/usr/bin/env python3
"""Create a COLMAP points3D.bin initialized by a dense XYZ/RGB/normal PLY."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from e57gs.surfels import write_colmap_points_from_ply


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(write_colmap_points_from_ply(args.input, args.output), indent=2))


if __name__ == "__main__":
    main()
