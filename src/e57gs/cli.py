"""Command-line entry point for the local E57 to 3DGS pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .e57io import read_metadata, verify_page_checksums
from .pipeline import prepare_dataset


def comma_separated_floats(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(part) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from error
    if not values:
        raise argparse.ArgumentTypeError("expected at least one pitch")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="e57gs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Read and validate E57 metadata")
    inspect_parser.add_argument("input", type=Path)
    inspect_parser.add_argument("--crc", action="store_true", help="Verify all E57 page CRC32C values")

    prepare_parser = subparsers.add_parser("prepare", help="Build a COLMAP training dataset")
    prepare_parser.add_argument("input", type=Path)
    prepare_parser.add_argument("output", type=Path)
    prepare_parser.add_argument("--size", type=int, default=2048)
    prepare_parser.add_argument("--fov", type=float, default=90.0)
    prepare_parser.add_argument(
        "--yaw-steps",
        type=int,
        default=4,
        help="number of evenly spaced horizontal views per pitch row",
    )
    prepare_parser.add_argument(
        "--pitches",
        type=comma_separated_floats,
        default=(-35.0, 0.0, 35.0),
        metavar="DEG,DEG,...",
        help="virtual camera pitch rows; use --pitches=-25,0,25 to avoid 20-degree polar caps",
    )
    prepare_parser.add_argument("--voxel-size", type=float, default=0.03)
    prepare_parser.add_argument("--polar-cap", type=float, default=20.0)
    prepare_parser.add_argument("--skip-crc", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "inspect":
        result = read_metadata(args.input).to_dict()
        if args.crc:
            result["crc"] = verify_page_checksums(args.input)
    elif args.command == "prepare":
        result = prepare_dataset(
            args.input,
            args.output,
            output_size=args.size,
            fov_deg=args.fov,
            num_steps_yaw=args.yaw_steps,
            pitches_deg=args.pitches,
            voxel_size_m=args.voxel_size,
            polar_cap_deg=args.polar_cap,
            verify_crc=not args.skip_crc,
        )
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
