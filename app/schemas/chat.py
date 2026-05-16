"""对话相关模型"""
from pydantic import BaseModel, Field


class ChatStreamRequest(BaseModel):
    conversation_id: str
    message: str = Field(min_length=1, max_length=10000)
