"""会话相关模型"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class ConversationCreate(BaseModel):
    title: str = Field(default="新对话", max_length=256)
    system_prompt: Optional[str] = None
    current_model: Optional[str] = None


class ConversationUpdate(BaseModel):
    title: Optional[str] = Field(default=None, max_length=256)
    status: Optional[str] = None
    summary: Optional[str] = None
    system_prompt: Optional[str] = None


class ConversationResponse(BaseModel):
    id: str
    user_id: str
    title: str
    current_model: Optional[str] = None
    summary: Optional[str] = None
    total_tokens: int = 0
    status: str
    created_at: datetime
    updated_at: datetime
