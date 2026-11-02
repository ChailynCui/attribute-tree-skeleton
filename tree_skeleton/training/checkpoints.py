# SPDX-License-Identifier: MIT
# Adapted research implementation; see README.md for license notices.
import re
from pathlib import Path
from typing import Iterable


_MILESTONE_PATTERNS = (
    re.compile(r"^model-(\d+)\.pt$"),
    re.compile(r"^line-transformer(?:\.ckpt)?[.-](\d+)\.pt$"),
)


def milestone_number(path: Path) -> int | None:
    for pattern in _MILESTONE_PATTERNS:
        match = pattern.fullmatch(path.name)
        if match is not None:
            return int(match.group(1))
    return None


def list_milestone_checkpoints(checkpoint_folder: str | Path) -> list[Path]:
    folder = Path(checkpoint_folder)
    if not folder.is_dir():
        return []

    checkpoints = [
        path for path in folder.iterdir() if path.is_file() and milestone_number(path) is not None
    ]
    return sorted(
        checkpoints,
        key=lambda path: (milestone_number(path), path.name),
    )


def plan_milestone_prune(
    checkpoint_folder: str | Path,
    keep_last: int = 1,
    eligible_paths: Iterable[Path] | None = None,
) -> list[Path]:
    if keep_last < 1:
        raise ValueError("keep_last must be at least 1")

    if eligible_paths is None:
        checkpoints = list_milestone_checkpoints(checkpoint_folder)
    else:
        checkpoints = sorted(
            (
                Path(path)
                for path in eligible_paths
                if Path(path).is_file() and milestone_number(Path(path)) is not None
            ),
            key=lambda path: (milestone_number(path), path.name),
        )

    return checkpoints[:-keep_last]


def prune_milestone_checkpoints(
    checkpoint_folder: str | Path,
    keep_last: int = 1,
) -> list[Path]:
    removed = plan_milestone_prune(checkpoint_folder, keep_last=keep_last)
    for path in removed:
        path.unlink()
    return removed
