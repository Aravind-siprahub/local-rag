"""Centralized password strength policy and reusable Pydantic field type."""
import re
from dataclasses import dataclass
from typing import Annotated, Final

from pydantic import AfterValidator, Field

# Printable ASCII special characters commonly accepted by password policies.
_SPECIAL_CHAR_RE: Final = re.compile(r"[!@#$%^&*(),.?\":{}|<>_\-+=\[\]\\;/`~]")

PASSWORD_MIN_LENGTH: Final = 8
PASSWORD_MAX_LENGTH: Final = 128
PASSWORD_EXAMPLE: Final = "SecurePass123!"

PASSWORD_DESCRIPTION: Final = (
    f"At least {PASSWORD_MIN_LENGTH} characters with uppercase, lowercase, "
    "digit, and special character."
)


@dataclass(frozen=True, slots=True)
class PasswordPolicy:
    """Password complexity rules — single source of truth for all schemas."""

    min_length: int = PASSWORD_MIN_LENGTH
    max_length: int = PASSWORD_MAX_LENGTH
    require_uppercase: bool = True
    require_lowercase: bool = True
    require_digit: bool = True
    require_special: bool = True

    def validate(self, value: str) -> str:
        if len(value) < self.min_length:
            raise ValueError(
                f"Password must be at least {self.min_length} characters long."
            )
        if len(value) > self.max_length:
            raise ValueError(
                f"Password must be at most {self.max_length} characters long."
            )
        if self.require_uppercase and not any(c.isupper() for c in value):
            raise ValueError("Password must contain at least one uppercase letter.")
        if self.require_lowercase and not any(c.islower() for c in value):
            raise ValueError("Password must contain at least one lowercase letter.")
        if self.require_digit and not any(c.isdigit() for c in value):
            raise ValueError("Password must contain at least one digit.")
        if self.require_special and not _SPECIAL_CHAR_RE.search(value):
            raise ValueError("Password must contain at least one special character.")
        return value


DEFAULT_PASSWORD_POLICY = PasswordPolicy()


def validate_password_strength(
    value: str,
    policy: PasswordPolicy = DEFAULT_PASSWORD_POLICY,
) -> str:
    """Validate a plaintext password against the configured policy."""
    return policy.validate(value)


PasswordField = Annotated[
    str,
    Field(
        min_length=PASSWORD_MIN_LENGTH,
        max_length=PASSWORD_MAX_LENGTH,
        description=PASSWORD_DESCRIPTION,
        examples=[PASSWORD_EXAMPLE],
        # OpenAPI 3.0 / Swagger UI reads singular `example`; Pydantic's
        # `examples` alone becomes a JSON Schema array Swagger often ignores.
        json_schema_extra={"example": PASSWORD_EXAMPLE},
    ),
    AfterValidator(validate_password_strength),
]
