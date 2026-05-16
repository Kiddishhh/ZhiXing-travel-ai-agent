"""用户路由：当前用户信息 + 长期画像"""
from fastapi import APIRouter, Depends
from psycopg_pool import AsyncConnectionPool

from app.schemas.user import UserResponse, UserProfileResponse
from app.api.v1.deps import get_current_user, get_db, get_memory_manager
from app.core.memory_store import MemoryStoreManager

router = APIRouter(prefix="/users", tags=["用户"])


@router.get("/me", response_model=UserResponse)
async def get_me(pool_user: tuple = Depends(get_db)):
    """获取当前用户信息"""
    pool, user_id = pool_user
    async with pool.connection() as conn:
        row = await conn.fetchrow(
            "SELECT id, username, email, avatar_url, role, created_at FROM users WHERE id = $1",
            user_id,
        )
    return dict(row)


@router.get("/me/profile", response_model=UserProfileResponse)
async def get_my_profile(
    pool_user: tuple = Depends(get_db),
    memory_mgr: MemoryStoreManager = Depends(get_memory_manager),
):
    """获取当前用户的长期旅行画像"""
    _, user_id = pool_user
    profile = await memory_mgr.get_profile(user_id)

    if profile is None:
        return UserProfileResponse(user_id=user_id)

    return UserProfileResponse(
        user_id=user_id,
        preferred_transport=profile.get("preferred_transport"),
        budget_level=profile.get("budget_level"),
        travel_styles=profile.get("travel_styles") or [],
        favorite_destinations=profile.get("favorite_destinations") or [],
        dietary_preferences=profile.get("dietary_preferences") or [],
        total_trips=profile.get("total_trips", 0),
        last_destination=profile.get("last_destination"),
        last_travel_date=str(profile["last_travel_date"]) if profile.get("last_travel_date") else None,
        extensions=profile.get("extensions") or {},
    )
