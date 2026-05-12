from __future__ import annotations

import math
from collections.abc import Iterable

import pandas as pd

try:
    from .recommender import build_recommender
    from .utils import PROCESSED_DIR
except ImportError:
    from recommender import build_recommender
    from utils import PROCESSED_DIR

POSITIVE_EVENTS = {"like", "complete", "replay", "share", "follow_show"}


def precision_at_k(recommended: Iterable[str], relevant: set[str], k: int) -> float:
    top_items = list(recommended)[:k]
    if not top_items:
        return 0.0
    return sum(item in relevant for item in top_items) / len(top_items)


def recall_at_k(recommended: Iterable[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top_items = list(recommended)[:k]
    return sum(item in relevant for item in top_items) / len(relevant)


def ndcg_at_k(recommended: Iterable[str], relevant: set[str], k: int) -> float:
    top_items = list(recommended)[:k]
    dcg = sum((1 / math.log2(rank + 2)) for rank, item in enumerate(top_items) if item in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1 / math.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


def catalog_coverage(recommendation_lists: Iterable[Iterable[str]], catalog: set[str]) -> float:
    if not catalog:
        return 0.0
    recommended_items = {item for recommendations in recommendation_lists for item in recommendations}
    return len(recommended_items & catalog) / len(catalog)


def positive_interactions(interactions: pd.DataFrame) -> pd.DataFrame:
    return interactions[
        interactions["event_type"].isin(POSITIVE_EVENTS) | (interactions["completion_rate"] >= 0.7)
    ].copy()


def evaluate(k: int = 10) -> dict[str, object]:
    recommender = build_recommender()
    interactions = pd.read_csv(PROCESSED_DIR / "interactions_processed.csv")
    content = pd.read_csv(PROCESSED_DIR / "content_processed.csv")
    positives = positive_interactions(interactions)

    precision_scores: list[float] = []
    recall_scores: list[float] = []
    ndcg_scores: list[float] = []
    recommendation_lists: list[list[str]] = []
    recommended_rows: list[dict[str, object]] = []

    for user_id, user_positive_events in positives.groupby("user_id"):
        relevant = set(user_positive_events["content_id"])
        recommendations = recommender.recommend_for_user(str(user_id), top_k=k, exclude_seen=False)
        recommended_ids = [item.content_id for item in recommendations]
        recommendation_lists.append(recommended_ids)
        recommended_rows.extend(item.to_dict() for item in recommendations)
        precision_scores.append(precision_at_k(recommended_ids, relevant, k))
        recall_scores.append(recall_at_k(recommended_ids, relevant, k))
        ndcg_scores.append(ndcg_at_k(recommended_ids, relevant, k))

    recommended_frame = pd.DataFrame(recommended_rows)
    catalog = set(content["content_id"])
    genre_coverage = 0.0
    tier_distribution: dict[str, float] = {}
    if not recommended_frame.empty:
        genre_coverage = recommended_frame["genre"].nunique() / max(content["genre"].nunique(), 1)
        tier_distribution = recommended_frame["tier"].value_counts(normalize=True).round(4).to_dict()

    return {
        f"precision@{k}": round(sum(precision_scores) / len(precision_scores), 4) if precision_scores else 0.0,
        f"recall@{k}": round(sum(recall_scores) / len(recall_scores), 4) if recall_scores else 0.0,
        f"ndcg@{k}": round(sum(ndcg_scores) / len(ndcg_scores), 4) if ndcg_scores else 0.0,
        "catalog_coverage": round(catalog_coverage(recommendation_lists, catalog), 4),
        "genre_coverage": round(genre_coverage, 4),
        "tier_distribution": tier_distribution,
        "evaluated_users": len(precision_scores),
    }


if __name__ == "__main__":
    print(evaluate(k=10))
