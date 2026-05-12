from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from .utils import PROCESSED_DIR, RAW_DIR, ensure_directories, join_text
except ImportError:
    from utils import PROCESSED_DIR, RAW_DIR, ensure_directories, join_text

CONTENT_COLUMNS = [
    "content_id",
    "show_name",
    "title",
    "content_type",
    "genre",
    "tags",
    "duration_seconds",
    "is_premium",
    "tier",
]

INTERACTIONS_COLUMNS = [
    "user_id",
    "content_id",
    "event_type",
    "listen_duration_sec",
    "content_duration_sec",
    "completion_rate",
    "timestamp",
    "device",
    "source",
]

EVENT_WEIGHTS = {
    "skip": -1.0,
    "play": 1.0,
    "like": 4.0,
    "complete": 4.5,
    "replay": 5.0,
    "share": 4.0,
    "follow_show": 5.0,
}

TEXT_COLUMNS = ["title", "show_name", "genre", "tags", "tier", "content_type"]


def load_csv(path: Path, required_columns: list[str]) -> pd.DataFrame:
    """Load a CSV file and validate that required columns exist."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV file: {path}")
    frame = pd.read_csv(path)
    validate_columns(frame, required_columns, path.name)
    return frame


def validate_columns(frame: pd.DataFrame, required_columns: list[str], name: str) -> None:
    """Raise a helpful error when the CSV schema is incomplete."""
    missing = sorted(set(required_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")


def create_sample_data() -> None:
    """Create Noice-inspired sample data only when raw CSV files are missing."""
    ensure_directories()
    interactions_path = RAW_DIR / "interactions.csv"
    content_path = RAW_DIR / "content.csv"

    if not content_path.exists():
        pd.DataFrame(
            [
                ["c_001", "Musuh Masyarakat", "E173: Kami Mendukung Skincare Abal-Abal!", "podcast", "komedi", "komedi satire masyarakat sosial", 3720, True, "vip"],
                ["c_002", "Scary Things", "Cerita Malam Jumat", "podcast", "horror", "horror misteri cerita seram", 3600, False, "free"],
                ["c_003", "TRIO KURNIA", "Obrolan Absurd Tengah Malam", "podcast", "komedi", "komedi obrolan pop culture", 3000, False, "free"],
                ["c_004", "Ghost Ranger", "Penjaga Rumah Tua", "audiobook", "horror", "ghost thriller supernatural", 4200, True, "premium"],
                ["c_005", "Noice Radio", "Morning Update", "radio", "news", "berita pagi aktual", 1800, False, "free"],
            ],
            columns=CONTENT_COLUMNS,
        ).to_csv(content_path, index=False)

    if not interactions_path.exists():
        pd.DataFrame(
            [
                ["u_001", "c_001", "like", 3100, 3720, 0.8333, "2026-01-01 21:00:00", "android", "detail_page"],
                ["u_001", "c_002", "complete", 3500, 3600, 0.9722, "2026-01-02 22:15:00", "ios", "search"],
                ["u_001", "c_005", "skip", 120, 1800, 0.0667, "2026-01-03 07:30:00", "web", "home"],
                ["u_002", "c_003", "share", 2400, 3000, 0.8000, "2026-01-04 20:10:00", "android", "share_link"],
                ["u_003", "c_004", "replay", 4100, 4200, 0.9762, "2026-01-05 23:40:00", "ios", "detail_page"],
            ],
            columns=INTERACTIONS_COLUMNS,
        ).to_csv(interactions_path, index=False)


def clean_text(value: object) -> str:
    """Normalize text values for metadata features."""
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def to_bool(value: object) -> bool:
    """Convert mixed CSV boolean values into bool."""
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y", "vip", "premium"}


def completion_bucket(rate: float) -> str:
    """Bucket completion rate into low, medium, and high listening intent."""
    if rate >= 0.7:
        return "high"
    if rate >= 0.3:
        return "medium"
    return "low"


def normalize_content(content: pd.DataFrame) -> pd.DataFrame:
    """Clean content metadata and build text fields for content similarity."""
    content = content.dropna(subset=["content_id", "title"]).drop_duplicates("content_id").copy()
    content["content_id"] = content["content_id"].astype(str).str.strip()
    for column in TEXT_COLUMNS:
        content[column] = content[column].map(clean_text)
    content["duration_seconds"] = pd.to_numeric(content["duration_seconds"], errors="coerce").fillna(0).astype(int)
    content["duration_minutes"] = content["duration_seconds"] / 60
    content["is_premium"] = content["is_premium"].map(to_bool)
    content["content_text"] = content.apply(
        lambda row: join_text([row.get("title"), row.get("show_name"), row.get("genre"), row.get("tags"), row.get("content_type"), row.get("tier")]),
        axis=1,
    )
    return content


def normalize_interactions(interactions: pd.DataFrame) -> pd.DataFrame:
    """Clean interactions and derive implicit feedback features."""
    interactions = interactions.dropna(subset=["user_id", "content_id"]).copy()
    interactions["user_id"] = interactions["user_id"].astype(str).str.strip()
    interactions["content_id"] = interactions["content_id"].astype(str).str.strip()
    interactions["event_type"] = interactions["event_type"].map(clean_text).replace("", "play")
    interactions["device"] = interactions["device"].map(clean_text)
    interactions["source"] = interactions["source"].map(clean_text)
    interactions["timestamp"] = pd.to_datetime(interactions["timestamp"], errors="coerce")

    numeric_columns = ["listen_duration_sec", "content_duration_sec", "completion_rate"]
    for column in numeric_columns:
        interactions[column] = pd.to_numeric(interactions[column], errors="coerce").fillna(0)

    missing_rate = interactions["completion_rate"] <= 0
    interactions.loc[missing_rate, "completion_rate"] = (
        interactions.loc[missing_rate, "listen_duration_sec"] / interactions.loc[missing_rate, "content_duration_sec"].replace(0, pd.NA)
    ).fillna(0)
    interactions["completion_rate"] = interactions["completion_rate"].clip(0, 1)
    interactions["listen_hour"] = interactions["timestamp"].dt.hour.fillna(0).astype(int)
    interactions["is_night_listener"] = (interactions["listen_hour"] >= 20) | (interactions["listen_hour"] <= 4)
    interactions["completion_bucket"] = interactions["completion_rate"].map(completion_bucket)

    event_score = interactions["event_type"].map(EVENT_WEIGHTS).fillna(1.0)
    source_bonus = interactions["source"].isin(["detail_page", "search"]).astype(float) * 0.5
    device_bonus = interactions["device"].isin(["android", "ios"]).astype(float) * 0.3
    interactions["implicit_score"] = (event_score + 3.0 * interactions["completion_rate"] + source_bonus + device_bonus).clip(lower=0)
    interactions["score"] = interactions["implicit_score"]
    return interactions


def build_user_item_matrix(interactions: pd.DataFrame) -> pd.DataFrame:
    """Create a user-item implicit feedback matrix."""
    return interactions.pivot_table(
        index="user_id",
        columns="content_id",
        values="implicit_score",
        aggfunc="sum",
        fill_value=0,
    )


def preprocess() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the Noice-inspired preprocessing pipeline."""
    ensure_directories()
    create_sample_data()

    content = normalize_content(load_csv(RAW_DIR / "content.csv", CONTENT_COLUMNS))
    interactions = normalize_interactions(load_csv(RAW_DIR / "interactions.csv", INTERACTIONS_COLUMNS))
    interactions = interactions[interactions["content_id"].isin(set(content["content_id"]))].copy()

    training_events = interactions.merge(
        content[
            [
                "content_id",
                "show_name",
                "title",
                "content_type",
                "genre",
                "tags",
                "tier",
                "duration_seconds",
                "is_premium",
                "duration_minutes",
                "content_text",
            ]
        ],
        on="content_id",
        how="left",
        suffixes=("", "_content"),
    )
    user_item_matrix = build_user_item_matrix(interactions)

    content.to_csv(PROCESSED_DIR / "content_processed.csv", index=False)
    interactions.to_csv(PROCESSED_DIR / "interactions_processed.csv", index=False)
    training_events.to_csv(PROCESSED_DIR / "training_events.csv", index=False)
    user_item_matrix.to_csv(PROCESSED_DIR / "user_item_matrix.csv")
    return interactions, content


if __name__ == "__main__":
    interactions_df, content_df = preprocess()
    print(f"Processed {len(interactions_df)} synthetic interactions and {len(content_df)} public catalog items.")
    print(f"Wrote processed files to {PROCESSED_DIR}")
