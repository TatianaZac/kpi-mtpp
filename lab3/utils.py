from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable


DEFAULT_SEED = 42


def make_rng(seed: int = DEFAULT_SEED) -> random.Random:
    return random.Random(seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0
