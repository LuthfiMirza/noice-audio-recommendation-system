from __future__ import annotations

import argparse
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from .data_pipeline import preprocess
    from .utils import MODELS_DIR, PROCESSED_DIR, ensure_directories, normalize_scores, top_k as select_top_k
except ImportError:
    from data_pipeline import preprocess
    from utils import MODELS_DIR, PROCESSED_DIR, ensure_directories, normalize_scores, top_k as select_top_k

MODEL_ARTIFACT_PATH = MODELS_DIR / "hybrid_recommender.pkl"
CONTENT_METADATA_COLUMNS = [
    "content_id",
    "title",
    "show_name",
    "content_type",
    "genre",
    "tier",
    "duration_seconds",
]


@dataclass
class Recommendation:
    """Serializable recommendation item returned by the model and API."""

    content_id: str
    title: str
    show_name: str
    content_type: str
    genre: str
    tier: str
    duration_seconds: int
    score: float
    reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class HybridRecommender:
    """Noice-inspired hybrid recommender for audio content discovery."""

    def __init__(self, cf_weight: float = 0.6, content_weight: float = 0.4) -> None:
        self.cf_weight = cf_weight
        self.content_weight = content_weight
        self.interactions = pd.DataFrame()
        self.content = pd.DataFrame()
        self.popularity = pd.DataFrame()
        self.item_similarity = None
        self.content_similarity = None
        self.tfidf_vectorizer: TfidfVectorizer | None = None
        self.content_index: dict[str, int] = {}
        self.user_ids: list[str] = []

    def fit(self, interactions: pd.DataFrame, content: pd.DataFrame) -> "HybridRecommender":
        """Fit collaborative, content-based, and popularity components."""
        self.interactions = interactions.copy()
        self.content = content.copy().reset_index(drop=True)
        self.content_index = {content_id: index for index, content_id in enumerate(self.content["content_id"])}
        self.user_ids = sorted(self.interactions["user_id"].dropna().unique().tolist())
        self._fit_item_cf()
        self._fit_content_similarity()
        self._fit_popularity()
        return self

    def recommend(self, user_id: str, top_n: int = 10, exclude_seen: bool = True) -> list[Recommendation]:
        """Backward-compatible alias for personalized recommendations."""
        return self.recommend_for_user(user_id, top_k=top_n, exclude_seen=exclude_seen)

    def recommend_for_user(self, user_id: str, top_k: int = 10, exclude_seen: bool = True) -> list[Recommendation]:
        """Return hybrid personalized recommendations for a known user."""
        if self.content.empty:
            return []
        if user_id not in set(self.user_ids):
            return self.recommend_trending(top_k=top_k)

        consumed = self._seen_content(user_id)
        if not consumed:
            return self.recommend_trending(top_k=top_k)

        scores = pd.DataFrame({"content_id": self.content["content_id"], "score": 0.0})
        cf_scores = self._similarity_scores(consumed, self.item_similarity)
        content_scores = self._similarity_scores(consumed, self.content_similarity)
        scores["score"] = self.cf_weight * normalize_scores(cf_scores) + self.content_weight * normalize_scores(content_scores)
        if exclude_seen:
            scores = scores[~scores["content_id"].isin(consumed)]
        user_genre = self._top_user_genre(user_id)
        reason = f"Based on your {user_genre} listening pattern" if user_genre else "Similar to shows you completed"
        return self._format_recommendations(select_top_k(scores, "score", top_k), reason=reason)

    def recommend_similar_content(self, content_id: str, top_k: int = 10) -> list[Recommendation]:
        """Return content similar to a given item using metadata similarity."""
        if content_id not in self.content_index or self.content_similarity is None:
            return self.recommend_trending(top_k=top_k)
        item_index = self.content_index[content_id]
        scores = pd.DataFrame({
            "content_id": self.content["content_id"],
            "score": self.content_similarity[item_index],
        })
        scores = scores[scores["content_id"] != content_id]
        genre = self.content.loc[item_index, "genre"]
        return self._format_recommendations(select_top_k(scores, "score", top_k), reason=f"Because it is similar to {genre} content")

    def recommend_trending(self, top_k: int = 10) -> list[Recommendation]:
        """Return popular content based on aggregate implicit feedback and event count."""
        if self.popularity.empty:
            scores = self.content[["content_id"]].assign(score=0.0)
        else:
            scores = self.popularity[["content_id", "score"]]
        return self._format_recommendations(select_top_k(scores, "score", top_k), reason="Popular among listeners")

    def recommend_by_genre(self, genre: str, top_k: int = 10) -> list[Recommendation]:
        """Return top content for a genre using popularity within that genre."""
        normalized_genre = str(genre).strip().lower()
        content_ids = self.content.loc[self.content["genre"].str.lower() == normalized_genre, "content_id"]
        if content_ids.empty:
            return []
        scores = self.popularity[self.popularity["content_id"].isin(content_ids)] if not self.popularity.empty else pd.DataFrame()
        if scores.empty:
            scores = pd.DataFrame({"content_id": content_ids, "score": 0.0})
        return self._format_recommendations(select_top_k(scores, "score", top_k), reason=f"Popular {normalized_genre} content")

    def response_mode_for_user(self, user_id: str) -> str:
        """Describe whether a request will use personalization or cold-start fallback."""
        return "personalized" if user_id in set(self.user_ids) and self._seen_content(user_id) else "cold_start_popular"

    def _fit_item_cf(self) -> None:
        if self.interactions.empty or self.content.empty:
            return
        users = {user_id: index for index, user_id in enumerate(self.interactions["user_id"].unique())}
        rows = self.interactions["user_id"].map(users)
        cols = self.interactions["content_id"].map(self.content_index)
        valid = rows.notna() & cols.notna()
        matrix = csr_matrix(
            (self.interactions.loc[valid, "implicit_score"], (rows[valid], cols[valid])),
            shape=(len(users), len(self.content_index)),
        )
        self.item_similarity = cosine_similarity(matrix.T)

    def _fit_content_similarity(self) -> None:
        if self.content.empty:
            return
        self.tfidf_vectorizer = TfidfVectorizer(stop_words=None, ngram_range=(1, 2), min_df=1)
        tfidf = self.tfidf_vectorizer.fit_transform(self.content["content_text"].fillna(""))
        self.content_similarity = cosine_similarity(tfidf)

    def _fit_popularity(self) -> None:
        if self.interactions.empty:
            self.popularity = pd.DataFrame(columns=["content_id", "score", "event_count"])
            return
        popularity = self.interactions.groupby("content_id", as_index=False).agg(
            implicit_score_sum=("implicit_score", "sum"),
            event_count=("event_type", "count"),
            avg_completion=("completion_rate", "mean"),
        )
        popularity["score"] = (
            normalize_scores(popularity["implicit_score_sum"]) * 0.7
            + normalize_scores(popularity["event_count"]) * 0.2
            + normalize_scores(popularity["avg_completion"]) * 0.1
        )
        self.popularity = popularity.sort_values("score", ascending=False).reset_index(drop=True)

    def _seen_content(self, user_id: str) -> list[str]:
        return self.interactions.loc[self.interactions["user_id"] == user_id, "content_id"].dropna().unique().tolist()

    def _similarity_scores(self, consumed: list[str], similarity_matrix) -> pd.Series:
        scores = pd.Series(0.0, index=self.content.index)
        if similarity_matrix is None:
            return scores
        consumed_indices = [self.content_index[item] for item in consumed if item in self.content_index]
        if not consumed_indices:
            return scores
        return pd.Series(similarity_matrix[consumed_indices].mean(axis=0), index=self.content.index)

    def _top_user_genre(self, user_id: str) -> str:
        user_events = self.interactions[self.interactions["user_id"] == user_id]
        if user_events.empty:
            return ""
        merged = user_events.merge(self.content[["content_id", "genre"]], on="content_id", how="left")
        if merged["genre"].dropna().empty:
            return ""
        return str(merged.groupby("genre")["implicit_score"].sum().sort_values(ascending=False).index[0])

    def _format_recommendations(self, frame: pd.DataFrame, reason: str) -> list[Recommendation]:
        if frame.empty:
            return []
        enriched = frame.merge(self.content[CONTENT_METADATA_COLUMNS], on="content_id", how="left")
        recommendations: list[Recommendation] = []
        for row in enriched.itertuples(index=False):
            recommendations.append(
                Recommendation(
                    content_id=str(row.content_id),
                    title=str(row.title),
                    show_name=str(row.show_name),
                    content_type=str(row.content_type),
                    genre=str(row.genre),
                    tier=str(row.tier),
                    duration_seconds=int(row.duration_seconds),
                    score=round(float(row.score), 6),
                    reason=reason,
                )
            )
        return recommendations


def load_processed_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load processed data, running preprocessing when outputs are missing."""
    interactions_path = PROCESSED_DIR / "interactions_processed.csv"
    content_path = PROCESSED_DIR / "content_processed.csv"
    if not interactions_path.exists() or not content_path.exists():
        return preprocess()
    return pd.read_csv(interactions_path), pd.read_csv(content_path)


def build_recommender() -> HybridRecommender:
    """Build a recommender from processed CSV data."""
    interactions, content = load_processed_data()
    return HybridRecommender().fit(interactions, content)


def save_recommender(recommender: HybridRecommender, path: Path = MODEL_ARTIFACT_PATH) -> None:
    """Persist a trained recommender artifact."""
    ensure_directories()
    with path.open("wb") as model_file:
        pickle.dump(recommender, model_file)


def load_recommender(path: Path = MODEL_ARTIFACT_PATH) -> HybridRecommender:
    """Load a persisted recommender artifact."""
    with path.open("rb") as model_file:
        return pickle.load(model_file)


def load_or_train_recommender(path: Path = MODEL_ARTIFACT_PATH) -> HybridRecommender:
    """Load a persisted recommender, or train and save one if unavailable/stale."""
    if path.exists():
        try:
            return load_recommender(path)
        except Exception:
            path.unlink(missing_ok=True)
    recommender = build_recommender()
    save_recommender(recommender, path)
    return recommender


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and query the Noice-inspired hybrid recommender.")
    parser.add_argument("--model", choices=["hybrid"], default="hybrid")
    parser.add_argument("--mode", choices=["user", "similar", "trending", "genre"], default="user")
    parser.add_argument("--user-id", default="u_001")
    parser.add_argument("--content-id", default="c_001")
    parser.add_argument("--genre", default="komedi")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--include-seen", action="store_true", help="Include content the user already interacted with.")
    parser.add_argument("--save-model", action="store_true", help="Persist trained recommender to models/.")
    args = parser.parse_args()

    recommender = build_recommender()
    if args.save_model:
        save_recommender(recommender)
        print(f"Saved model artifact to {MODEL_ARTIFACT_PATH}")

    if args.mode == "similar":
        recommendations = recommender.recommend_similar_content(args.content_id, args.top_k)
    elif args.mode == "trending":
        recommendations = recommender.recommend_trending(args.top_k)
    elif args.mode == "genre":
        recommendations = recommender.recommend_by_genre(args.genre, args.top_k)
    else:
        recommendations = recommender.recommend_for_user(args.user_id, args.top_k, exclude_seen=not args.include_seen)

    for recommendation in recommendations:
        print(recommendation.to_dict())


if __name__ == "__main__":
    main()
