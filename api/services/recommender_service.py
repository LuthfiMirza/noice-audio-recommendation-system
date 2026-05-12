from __future__ import annotations

from functools import lru_cache

from src.recommender import HybridRecommender, load_or_train_recommender


@lru_cache(maxsize=1)
def get_recommender() -> HybridRecommender:
    return load_or_train_recommender()
