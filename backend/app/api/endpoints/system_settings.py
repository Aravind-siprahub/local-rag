"""System setting endpoints.

Uses PUT (not POST) for `/system-settings/{key}`: settings are addressed by
a caller-chosen key, and "create if absent, replace if present" is exactly
what PUT means in REST — matching `SystemSettingService.set_setting`'s
upsert semantics.
"""
from fastapi import APIRouter, Depends, status

from app.api.dependencies import PaginationParams, get_system_setting_service
from app.schemas.actions import SystemSettingUpsertRequest
from app.schemas.system_setting import SystemSettingListResponse, SystemSettingResponse
from app.services.system_setting_service import SystemSettingService

router = APIRouter(prefix="/system-settings", tags=["System Settings"])


@router.put("/{key}", response_model=SystemSettingResponse, summary="Create or replace a setting")
async def upsert_system_setting(
    key: str,
    payload: SystemSettingUpsertRequest,
    service: SystemSettingService = Depends(get_system_setting_service),
) -> SystemSettingResponse:
    setting = await service.set_setting(
        key=key, value=payload.value, description=payload.description, updated_by=payload.updated_by
    )
    return SystemSettingResponse.model_validate(setting)


@router.get("", response_model=SystemSettingListResponse, summary="List all settings")
async def list_system_settings(
    pagination: PaginationParams = Depends(),
    service: SystemSettingService = Depends(get_system_setting_service),
) -> SystemSettingListResponse:
    settings = await service.list(limit=pagination.limit, offset=pagination.offset)
    total = await service.count()
    return SystemSettingListResponse(
        items=[SystemSettingResponse.model_validate(s) for s in settings],
        total=total,
        limit=pagination.limit,
        offset=pagination.offset,
    )


@router.get("/{key}", response_model=SystemSettingResponse, summary="Get a setting by key")
async def get_system_setting(
    key: str, service: SystemSettingService = Depends(get_system_setting_service)
) -> SystemSettingResponse:
    setting = await service.get(key)
    return SystemSettingResponse.model_validate(setting)


@router.delete("/{key}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a setting")
async def delete_system_setting(
    key: str, service: SystemSettingService = Depends(get_system_setting_service)
) -> None:
    await service.delete(key)
