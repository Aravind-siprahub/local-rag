"""Schemas for `app.models.system_setting.SystemSetting`.

Doesn't use `TimestampSchema`/`CreatedAtSchema` from `common.py`: this is the
one table with `updated_at` but no `created_at` and no surrogate UUID key
(the primary key is `key: str`), so it gets its own minimal shape.
"""
import uuid
from datetime import datetime
from typing import Annotated, Any

from pydantic import BaseModel, Field

from app.schemas.common import ORMModel, PaginatedResponse


class SystemSettingBase(BaseModel):
    value: dict[str, Any]
    description: str | None = None


class SystemSettingCreate(SystemSettingBase):
    key: Annotated[str, Field(min_length=1, max_length=255)]


class SystemSettingUpdate(BaseModel):
    value: dict[str, Any] | None = None
    description: str | None = None
    updated_by: uuid.UUID | None = None


class SystemSettingResponse(SystemSettingBase, ORMModel):
    key: str
    updated_by: uuid.UUID | None = None
    updated_at: datetime


SystemSettingListResponse = PaginatedResponse[SystemSettingResponse]
