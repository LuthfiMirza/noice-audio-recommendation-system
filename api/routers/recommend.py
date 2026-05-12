from fastapi import APIRouter, HTTPException

from api.schemas import RecommendationItem, RecommendationRequest, RecommendationResponse
from api.services.recommender_service import get_recommender

router = APIRouter(tags=["recommendations"])


@router.post("/recommend", response_model=RecommendationResponse)
def recommend_content(request: RecommendationRequest) -> RecommendationResponse:
    try:
        recommender = get_recommender()
        mode = recommender.response_mode_for_user(request.user_id)
        recommendations = recommender.recommend_for_user(
            request.user_id,
            top_k=request.top_n,
            exclude_seen=request.exclude_seen,
        )
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=[RecommendationItem(**item.to_dict()) for item in recommendations],
            mode=mode,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
