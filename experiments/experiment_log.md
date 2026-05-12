# Experiment Log

## Noice-Inspired Hybrid Recommender

### Data

- `content.csv`: public catalog-style Noice-inspired metadata for portfolio/demo usage.
- `interactions.csv`: synthetic listening events from simulated users.
- No private/internal Noice user data is used.

### Preprocessing

- Validates raw CSV schemas.
- Converts event behavior into `implicit_score`.
- Adds temporal and behavior features such as `listen_hour`, `is_night_listener`, and `completion_bucket`.
- Builds `content_text` from title, show name, genre, tags, content type, and tier.

### Model

| Component | Description |
|---|---|
| Item-based CF | Uses user-item implicit feedback matrix |
| Content-based | Uses TF-IDF cosine similarity over `content_text` |
| Hybrid blend | `0.6 * collaborative_score + 0.4 * content_score` |
| Cold-start | Uses aggregate popularity/trending scores |

### Evaluation Plan

Current evaluator supports:

- Precision@K
- Recall@K
- nDCG@K
- Catalog coverage
- Genre coverage
- Tier distribution

### Next Experiments

- Add temporal train/test split.
- Compare content-only vs CF-only vs hybrid rankings.
- Tune hybrid alpha from `0.3` to `0.8`.
- Evaluate separate cold-start user and cold-start item scenarios.
- Add diversity controls to avoid over-recommending a single show or tier.
