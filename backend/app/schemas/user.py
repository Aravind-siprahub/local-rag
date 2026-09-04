"""Schemas for `app.models.user.User`."""
import uuid
from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from app.models.enums import UserRole
from app.schemas.common import ORMModel, PaginatedResponse, TimestampSchema
from app.schemas.openapi_examples import (
    EMAIL_EXAMPLE,
    FULL_NAME_EXAMPLE,
    USER_CREATE_OPENAPI_EXAMPLE,
)
from app.schemas.validators.password import PasswordField


class UserBase(BaseModel):
    """Fields common to creating and reading a user."""

    email: Annotated[
        EmailStr,
        Field(examples=[EMAIL_EXAMPLE], json_schema_extra={"example": EMAIL_EXAMPLE}),
    ]
    full_name: Annotated[
        str | None,
        Field(
            default=None,
            max_length=255,
            examples=[FULL_NAME_EXAMPLE],
            json_schema_extra={"example": FULL_NAME_EXAMPLE},
        ),
    ]


class UserCreate(UserBase):
    password: PasswordField
    role: UserRole = UserRole.MEMBER

    model_config = ConfigDict(json_schema_extra={"example": USER_CREATE_OPENAPI_EXAMPLE})


class UserUpdate(BaseModel):
    """All fields optional — partial update. Password changes are
    deliberately excluded here; that belongs behind a dedicated
    auth/security flow, not a generic field update.
    """

    email: EmailStr | None = None
    full_name: Annotated[str | None, Field(default=None, max_length=255)]
    role: UserRole | None = None
    is_active: bool | None = None
    is_verified: bool | None = None


class UserResponse(UserBase, TimestampSchema, ORMModel):
    id: uuid.UUID
    role: UserRole
    is_active: bool
    is_verified: bool
    is_2fa_enabled: bool = False
    last_login_at: datetime | None = None


UserListResponse = PaginatedResponse[UserResponse]


# --- 2FA Schemas --------------------------------------------------------------

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    full_name: str | None = None


class TwoFactorSetupResponse(BaseModel):
    totp_secret: str
    provisioning_uri: str
    qr_code_data_url: str
    backup_codes: list[str]
    temp_token: str | None = None
    message: str = "Scan the QR code with Google Authenticator or Microsoft Authenticator, save your recovery codes, and submit the 6-digit verification code."


class TwoFactorVerifyRequest(BaseModel):
    temp_token: str | None = None
    code: str  # Can be 6-digit TOTP or recovery code (e.g. XXXX-XXXX)
    is_backup_code: bool = False


class TwoFactorChallengeResponse(BaseModel):
    requires_2fa: bool = True
    requires_2fa_setup: bool = False
    temp_token: str
    message: str = "Please provide your 6-digit authenticator code or recovery code to complete sign in."


# --- Email Verification Schemas -----------------------------------------------

class RegistrationResponse(BaseModel):
    status: str = "verification_required"
    email: str
    message: str = "Verification code sent to your email. Please verify to complete registration."


class VerifyEmailRequest(BaseModel):
    email: EmailStr
    code: str


class ResendVerificationRequest(BaseModel):
    email: EmailStr


class VerifyEmailResponse(BaseModel):
    status: str = "verified"
    message: str = "Email successfully verified."
    access_token: str | None = None
    token_type: str = "bearer"
    user: UserResponse | None = None


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ForgotPasswordResponse(BaseModel):
    status: str = "sent"
    message: str = "If an account exists with this email, a 6-digit password reset code has been sent."


class ResetPasswordRequest(BaseModel):
    email: EmailStr
    code: str
    new_password: PasswordField


class ResetPasswordResponse(BaseModel):
    status: str = "success"
    message: str = "Password has been successfully reset. Please log in with your new password."

