"""Business logic for `app.models.system_setting.SystemSetting`.

Primary key is `key: str`, not a UUID — hence `BaseService[SystemSetting, str, ...]`.
"""
import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.system_setting import SystemSetting
from app.repositories.system_setting_repository import SystemSettingRepository
from app.services.base_service import BaseService


class SystemSettingService(BaseService[SystemSetting, str, SystemSettingRepository]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, SystemSettingRepository(session))

    async def set_setting(
        self,
        *,
        key: str,
        value: dict[str, Any],
        description: str | None = None,
        updated_by: uuid.UUID | None = None,
    ) -> SystemSetting:
        """Upsert: settings are addressed by a human-chosen key the caller
        may or may not know already exists, so "set" (create-or-replace) is
        the natural operation here — unlike every other domain, where
        create vs. update are meaningfully different actions.
        """
        existing = await self.get_optional(key)
        if existing is None:
            return await self.create(key=key, value=value, description=description, updated_by=updated_by)

        return await self.update(key, value=value, description=description, updated_by=updated_by)
