"""消息相关模型"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class MessageResponse(BaseModel):
    id: str
    conversation_id: str
    role: str
    content: str
    content_type: str = "text"
    token_count: int = 0
    feedback: int = 0
    is_error: bool = False
    metadata: dict = {}
    created_at: datetime
