from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.config import ALLOWED_ENTITY_TYPES


class ChatMessageRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = None
    user_id: str = "user"
    user_name: str = Field(min_length=1, max_length=120, default="User")
    prompt_profile: str = Field(min_length=1, max_length=80, default="hotel_customer_service")


class Triplet(BaseModel):
    subject: str = Field(min_length=1)
    subject_type: str
    relation: str = Field(min_length=1)
    object: str = Field(min_length=1)
    object_type: str
    confidence: float = Field(default=0.75, ge=0.0, le=1.0)

    @field_validator("subject_type")
    @classmethod
    def validate_subject_type(cls, value: str) -> str:
        normalized = value.strip()
        if normalized not in ALLOWED_ENTITY_TYPES:
            raise ValueError(f"subject_type must be one of {sorted(ALLOWED_ENTITY_TYPES)}")
        return normalized

    @field_validator("object_type")
    @classmethod
    def validate_object_type(cls, value: str) -> str:
        normalized = value.strip()
        if normalized not in ALLOWED_ENTITY_TYPES:
            raise ValueError(f"object_type must be one of {sorted(ALLOWED_ENTITY_TYPES)}")
        return normalized


class ChatMessageResponse(BaseModel):
    session_id: str
    user_name: str
    prompt_profile: str
    assistant_message: str
    extracted_triplets: list[Triplet]
    stored_triplets_count: int


class ChatHistoryItem(BaseModel):
    id: int
    session_id: str
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime


class ChatSessionCreateRequest(BaseModel):
    user_name: str = Field(min_length=1, max_length=120)


class ChatSessionSummary(BaseModel):
    id: str
    user_id: str
    user_name: str
    title: str
    created_at: datetime
    last_message_at: datetime | None = None


class EntityItem(BaseModel):
    name: str
    normalized_name: str
    entity_type: str


class RelationItem(BaseModel):
    subject: str
    subject_type: str
    relation: str
    object: str
    object_type: str
    mentions_count: int
    created_at: str | None = None
    updated_at: str | None = None
    last_confidence: float | None = None


class GraphRecentItem(BaseModel):
    subject: str
    relation: str
    object: str
    source_message_id: str | None = None
    updated_at: str | None = None
