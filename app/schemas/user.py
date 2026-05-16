"""用户相关模型"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    avatar_url: Optional[str] = None
    role: str
    created_at: datetime


class UserProfileResponse(BaseModel):
    user_id: str
    preferred_transport: Optional[str] = None
    budget_level: Optional[str] = None
    travel_styles: list[str] = []
    favorite_destinations: list[str] = []
    dietary_preferences: list[str] = []
    total_trips: int = 0
    last_destination: Optional[str] = None
    last_travel_date: Optional[str] = None
    extensions: dict = {}
