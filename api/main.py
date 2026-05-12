from fastapi import FastAPI

from api.routers.recommend import router as recommend_router

app = FastAPI(
    title="Noice-Inspired Audio Recommendation System",
    description="Hybrid audio recommender using public catalog metadata and synthetic listening events.",
    version="0.1.0",
)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(recommend_router)
