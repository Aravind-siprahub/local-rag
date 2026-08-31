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
    from app.core.config import get_settings
    conf = get_settings()

    settings_list = await service.list(limit=pagination.limit, offset=pagination.offset)
    existing_keys = {s.key: s for s in settings_list}

    default_map = {
        "CHAT_MODEL": {"val": conf.OLLAMA_MODEL or conf.CHAT_MODEL},
        "LLM_TEMPERATURE": {"val": conf.LLM_TEMPERATURE},
        "MAX_CONTEXT_TOKENS": {"val": conf.OLLAMA_NUM_CTX},
        "LLM_TIMEOUT": {"val": conf.LLM_TIMEOUT_SECONDS},
        "TOP_K": {"val": conf.TOP_K},
        "FINAL_CONTEXT": {"val": conf.FINAL_CONTEXT},
        "SIMILARITY_THRESHOLD": {"val": conf.SIMILARITY_THRESHOLD},
        "CHUNK_SIZE": {"val": conf.CHUNK_SIZE},
        "CHUNK_OVERLAP": {"val": conf.CHUNK_OVERLAP},
        "EMBEDDING_MODEL": {"val": conf.EMBEDDING_MODEL},
        "VECTOR_DIMENSIONS": {"val": conf.EMBEDDING_DIMENSIONS},
    }

    updated = False
    for key, def_val in default_map.items():
        if key not in existing_keys:
            setting = await service.set_setting(key=key, value=def_val, description=f"Active {key} setting")
            settings_list.append(setting)
            existing_keys[key] = setting
            updated = True
        elif key == "CHAT_MODEL" and existing_keys[key].value.get("val") == "qwen3:4b":
            setting = await service.set_setting(key=key, value={"val": "qwen3:8b"}, description="Active CHAT_MODEL setting")
            existing_keys[key].value = {"val": "qwen3:8b"}
            updated = True

    total = len(settings_list)
    return SystemSettingListResponse(
        items=[SystemSettingResponse.model_validate(s) for s in settings_list],
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
