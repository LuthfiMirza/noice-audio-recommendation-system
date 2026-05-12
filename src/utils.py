from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


def ensure_directories() -> None:
    for directory in (RAW_DIR, PROCESSED_DIR, MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def normalize_scores(scores: pd.Series) -> pd.Series:
    if scores.empty:
        return scores
    min_score = scores.min()
    max_score = scores.max()
    if max_score == min_score:
        return pd.Series(1.0, index=scores.index)
    return (scores - min_score) / (max_score - min_score)


def top_k(frame: pd.DataFrame, score_column: str, k: int) -> pd.DataFrame:
    return frame.sort_values(score_column, ascending=False).head(k).reset_index(drop=True)


def join_text(values: Iterable[object]) -> str:
    return " ".join(str(value).strip() for value in values if pd.notna(value) and str(value).strip())
