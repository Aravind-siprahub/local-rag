"""Tests for centralized password validation."""
import pytest
from pydantic import ValidationError as PydanticValidationError

from app.schemas.user import UserCreate
from app.schemas.validators.password import (
    PASSWORD_EXAMPLE,
    PasswordPolicy,
    validate_password_strength,
)


class TestValidatePasswordStrength:
    def test_accepts_production_example(self) -> None:
        assert validate_password_strength(PASSWORD_EXAMPLE) == PASSWORD_EXAMPLE

    def test_rejects_too_short(self) -> None:
        with pytest.raises(ValueError, match="at least 8 characters"):
            validate_password_strength("Ab1!")

    def test_rejects_missing_uppercase(self) -> None:
        with pytest.raises(ValueError, match="uppercase"):
            validate_password_strength("securepass123!")

    def test_rejects_missing_lowercase(self) -> None:
        with pytest.raises(ValueError, match="lowercase"):
            validate_password_strength("SECUREPASS123!")

    def test_rejects_missing_digit(self) -> None:
        with pytest.raises(ValueError, match="digit"):
            validate_password_strength("SecurePass!")

    def test_rejects_missing_special(self) -> None:
        with pytest.raises(ValueError, match="special character"):
            validate_password_strength("SecurePass123")

    def test_optional_special_when_disabled(self) -> None:
        policy = PasswordPolicy(require_special=False)
        assert validate_password_strength("SecurePass123", policy=policy) == "SecurePass123"


class TestUserCreateSchema:
    def test_openapi_example_is_valid(self) -> None:
        user = UserCreate(
            email="john.doe@example.com",
            full_name="John Doe",
            password=PASSWORD_EXAMPLE,
            role="member",
        )
        assert user.password == PASSWORD_EXAMPLE

    def test_rejects_swagger_string_password(self) -> None:
        with pytest.raises(PydanticValidationError):
            UserCreate(
                email="john.doe@example.com",
                full_name="John Doe",
                password="string",
            )

    def test_json_schema_contains_valid_password_example(self) -> None:
        schema = UserCreate.model_json_schema()
        assert schema["example"]["password"] == PASSWORD_EXAMPLE
        assert schema["properties"]["password"]["examples"] == [PASSWORD_EXAMPLE]
