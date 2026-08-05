"""Data access for `app.models.system_setting.SystemSetting`.

Primary key is `key: str`, not a UUID — hence `BaseRepository[SystemSetting, str]`.
"""
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.system_setting import SystemSetting
from app.repositories.base_repository import BaseRepository


class SystemSettingRepository(BaseRepository[SystemSetting, str]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, SystemSetting)

    async def get_by_key(self, key: str) -> SystemSetting | None:
        """Alias of `get()` — `key` reads clearer than `id_` at call sites
        for this particular table.
        """
        return await self.get(key)
