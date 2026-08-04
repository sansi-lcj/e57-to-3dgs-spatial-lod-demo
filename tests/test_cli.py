import argparse

import pytest

from e57gs.cli import build_parser, comma_separated_floats


def test_pitch_rows_can_be_overridden() -> None:
    args = build_parser().parse_args(
        ["prepare", "input.e57", "output", "--pitches=-25,0,25"]
    )
    assert args.pitches == (-25.0, 0.0, 25.0)


def test_yaw_steps_can_be_overridden() -> None:
    args = build_parser().parse_args(
        ["prepare", "input.e57", "output", "--yaw-steps", "5"]
    )
    assert args.yaw_steps == 5


def test_pitch_rows_reject_non_numbers() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        comma_separated_floats("-25,nope,25")
