"""Deterministic dataset split generation for frame-based experiments.

The helpers in this module intentionally avoid Nerfstudio, torch, and other
heavy runtime dependencies. They operate on frame identifiers and common
manifest formats so split files can be generated before any ML stack is
installed.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


DEFAULT_IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".exr",
)

FRAME_ID_KEYS = (
    "file_path",
    "image_path",
    "rgb_path",
    "path",
    "frame_id",
    "id",
    "name",
)


@dataclass(frozen=True)
class SplitPlan:
    """A full split plan containing one split per train budget."""

    scene: str | None
    seed: int
    total_frames: int
    val_count: int
    test_count: int
    train_counts: tuple[int, ...]
    splits: Mapping[str, Mapping[str, list[str]]]

    def to_dict(self) -> dict[str, object]:
        return {
            "scene": self.scene,
            "seed": self.seed,
            "total_frames": self.total_frames,
            "val_count": self.val_count,
            "test_count": self.test_count,
            "train_counts": list(self.train_counts),
            "splits": dict(self.splits),
        }


def load_frame_ids(
    source: str | Path,
    image_extensions: Sequence[str] = DEFAULT_IMAGE_EXTENSIONS,
) -> list[str]:
    """Load frame identifiers from a directory, JSON manifest, CSV, or text file.

    Supported JSON shapes:
    - ``["frame_000.png", ...]``
    - ``{"frames": [{"file_path": "..."}, ...]}``
    - ``{"images": [{"path": "..."}, ...]}``
    - ``{"frame_ids": ["frame_000", ...]}``

    Directories are scanned recursively for files matching ``image_extensions``
    and returned as POSIX-style paths relative to the directory.
    """

    path = Path(source)
    if path.is_dir():
        suffixes = {ext.lower() for ext in image_extensions}
        frames = [
            file.relative_to(path).as_posix()
            for file in path.rglob("*")
            if file.is_file() and file.suffix.lower() in suffixes
        ]
        return sorted(frames)

    if not path.exists():
        raise FileNotFoundError(f"Frame source does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return _frame_ids_from_json(payload)

    if suffix == ".csv":
        return _frame_ids_from_csv(path)

    return _frame_ids_from_text(path)


def generate_split_plan(
    frame_ids: Iterable[str],
    train_counts: Sequence[int],
    *,
    scene: str | None = None,
    seed: int = 0,
    val_fraction: float = 0.1,
    test_fraction: float = 0.2,
    val_count: int | None = None,
    test_count: int | None = None,
    shuffle: bool = True,
) -> SplitPlan:
    """Generate deterministic train/validation/test splits for train budgets.

    Validation and test frames are fixed across every budget. Training frames
    are prefix-nested, so the ``25``-frame split is a subset of the ``50``-frame
    split when both budgets are requested.
    """

    frames = [str(frame) for frame in frame_ids]
    _validate_unique(frames)
    if not frames:
        raise ValueError("At least one frame is required.")
    if not train_counts:
        raise ValueError("At least one train count is required.")

    counts = tuple(sorted({int(count) for count in train_counts}))
    if any(count <= 0 for count in counts):
        raise ValueError("Train counts must be positive integers.")

    total = len(frames)
    resolved_val_count = _resolve_holdout_count(
        explicit_count=val_count,
        fraction=val_fraction,
        total=total,
        name="val",
    )
    resolved_test_count = _resolve_holdout_count(
        explicit_count=test_count,
        fraction=test_fraction,
        total=total,
        name="test",
    )

    if resolved_val_count + resolved_test_count >= total:
        raise ValueError(
            "Validation and test holdouts leave no frames for training: "
            f"val={resolved_val_count}, test={resolved_test_count}, total={total}"
        )

    shuffled = list(frames)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(shuffled)

    test = shuffled[:resolved_test_count]
    val = shuffled[resolved_test_count : resolved_test_count + resolved_val_count]
    train_pool = shuffled[resolved_test_count + resolved_val_count :]

    max_count = counts[-1]
    if max_count > len(train_pool):
        raise ValueError(
            f"Requested train count {max_count} exceeds available training pool "
            f"of {len(train_pool)} frames after holdouts."
        )

    original_order = {frame: index for index, frame in enumerate(frames)}
    splits: dict[str, Mapping[str, list[str]]] = {}
    ordered_val = _order_like_input(val, original_order)
    ordered_test = _order_like_input(test, original_order)
    for count in counts:
        train = train_pool[:count]
        splits[str(count)] = {
            "train": _order_like_input(train, original_order),
            "val": ordered_val,
            "test": ordered_test,
        }

    return SplitPlan(
        scene=scene,
        seed=seed,
        total_frames=total,
        val_count=resolved_val_count,
        test_count=resolved_test_count,
        train_counts=counts,
        splits=splits,
    )


def write_split_plan(plan: SplitPlan, output_path: str | Path) -> None:
    """Write a split plan as stable, human-readable JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(plan.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _frame_ids_from_json(payload: object) -> list[str]:
    if isinstance(payload, list):
        return [_frame_id_from_item(item) for item in payload]

    if not isinstance(payload, dict):
        raise ValueError("JSON frame manifest must be a list or object.")

    for key in ("frames", "images", "frame_ids"):
        value = payload.get(key)
        if value is not None:
            if not isinstance(value, list):
                raise ValueError(f"JSON manifest key '{key}' must contain a list.")
            return [_frame_id_from_item(item) for item in value]

    raise ValueError(
        "JSON frame manifest must contain one of: frames, images, frame_ids."
    )


def _frame_id_from_item(item: object) -> str:
    if isinstance(item, (str, int)):
        return str(item)
    if isinstance(item, dict):
        for key in FRAME_ID_KEYS:
            value = item.get(key)
            if value is not None:
                return str(value)
        raise ValueError(
            f"Frame item is missing one of the supported id keys: {FRAME_ID_KEYS}"
        )
    raise ValueError(f"Unsupported frame item type: {type(item).__name__}")


def _frame_ids_from_csv(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        sample = handle.read(2048)
        handle.seek(0)
        try:
            has_header = csv.Sniffer().has_header(sample) if sample.strip() else False
        except csv.Error:
            has_header = False
        if has_header:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return []
            key = next(
                (field for field in FRAME_ID_KEYS if field in reader.fieldnames),
                reader.fieldnames[0],
            )
            return [str(row[key]) for row in reader if row.get(key)]

        reader = csv.reader(handle)
        return [row[0].strip() for row in reader if row and row[0].strip()]


def _frame_ids_from_text(path: Path) -> list[str]:
    frames: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            frames.append(stripped)
    return frames


def _resolve_holdout_count(
    *,
    explicit_count: int | None,
    fraction: float,
    total: int,
    name: str,
) -> int:
    if explicit_count is not None:
        count = int(explicit_count)
    else:
        if fraction < 0:
            raise ValueError(f"{name}_fraction must be non-negative.")
        count = int(round(total * fraction))

    if count < 0:
        raise ValueError(f"{name}_count must be non-negative.")
    return count


def _validate_unique(frames: Sequence[str]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for frame in frames:
        if frame in seen:
            duplicates.append(frame)
        seen.add(frame)
    if duplicates:
        sample = ", ".join(duplicates[:5])
        raise ValueError(f"Frame identifiers must be unique. Duplicates: {sample}")


def _order_like_input(
    frames: Iterable[str],
    original_order: Mapping[str, int],
) -> list[str]:
    return sorted(frames, key=lambda frame: original_order[frame])
