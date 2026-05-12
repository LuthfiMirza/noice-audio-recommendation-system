from pydantic import BaseModel, Field


class RecommendationRequest(BaseModel):
    user_id: str = Field(..., examples=["u_001"])
    top_n: int = Field(10, ge=1, le=100)
    exclude_seen: bool = True


class RecommendationItem(BaseModel):
    content_id: str
    title: str
    show_name: str
    content_type: str
    genre: str
    tier: str
    duration_seconds: int
    score: float
    reason: str


class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: list[RecommendationItem]
    mode: str
