"""Request/response models.

JSON uses camelCase to match the existing React components. Every inbound
field has explicit length and format limits; unknown fields are rejected.
"""
import re
from datetime import date
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator
from pydantic.alias_generators import to_camel

SAFE_TEXT = re.compile(r"^[^\x00-\x08\x0b\x0c\x0e-\x1f\x7f]*$")  # no control characters


class ApiModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid",
                              str_strip_whitespace=True)


def _clean(v: str | None) -> str | None:
    if v is not None and not SAFE_TEXT.match(v):
        raise ValueError("contains control characters")
    return v


def _validate_crops(v: list[str] | None) -> list[str] | None:
    if v is None:
        return v
    cleaned = []
    for c in v:
        c = c.strip()
        if not (1 <= len(c) <= 40) or not re.match(r"^[\w ()'-]+$", c):
            raise ValueError("invalid crop name")
        cleaned.append(c)
    return list(dict.fromkeys(cleaned))


# --- Auth -------------------------------------------------------------------

class RegisterRequest(ApiModel):
    username: str = Field(min_length=3, max_length=40, pattern=r"^[\w .'-]+$")
    email: EmailStr
    phone_number: str = Field(pattern=r"^(\+91[- ]?)?[6-9]\d{9}$")
    password: str = Field(min_length=1, max_length=128)
    # Sensitive identifiers: optional, encrypted at rest, only ever returned masked.
    aadhar_id: str | None = Field(default=None, pattern=r"^\d{4}-\d{4}-\d{4}$")
    pan_card_number: str | None = Field(default=None, pattern=r"^[A-Z]{5}[0-9]{4}[A-Z]$")
    farm_id: str | None = Field(default=None, pattern=r"^[A-Z]{0,4}\d{6,12}$")
    preferred_crops: list[str] = Field(default_factory=list, max_length=30)
    user_type: Literal["Farmer", "Home Gardener"] = "Farmer"

    _crops = field_validator("preferred_crops")(lambda cls, v: _validate_crops(v))

    @field_validator("aadhar_id")
    @classmethod
    def _aadhaar(cls, v):
        if v and re.search(r"(\d)\1{3}", v.replace("-", "")):
            raise ValueError("must not contain 4 repeated digits")
        return v


class LoginRequest(ApiModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class UserOut(ApiModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
    id: str
    username: str
    email: str
    phone_number: str
    user_type: str
    role: str
    preferred_crops: list[str] = []
    farm_id: str | None = None
    aadhar_id: str | None = None        # masked, e.g. XXXX-XXXX-9087
    pan_card_number: str | None = None  # masked, e.g. ABXXXXXX4F
    created_at: str
    disabled: bool = False


class TokenResponse(ApiModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserOut


class ProfileUpdate(ApiModel):
    username: str | None = Field(default=None, min_length=3, max_length=40, pattern=r"^[\w .'-]+$")
    phone_number: str | None = Field(default=None, pattern=r"^(\+91[- ]?)?[6-9]\d{9}$")
    farm_id: str | None = Field(default=None, pattern=r"^[A-Z]{0,4}\d{6,12}$")
    preferred_crops: list[str] | None = Field(default=None, max_length=30)

    _crops = field_validator("preferred_crops")(lambda cls, v: _validate_crops(v))


class PasswordChange(ApiModel):
    current_password: str = Field(min_length=1, max_length=128)
    new_password: str = Field(min_length=1, max_length=128)


class AdminUserUpdate(ApiModel):
    role: Literal["user", "admin"] | None = None
    disabled: bool | None = None


# --- Inventory --------------------------------------------------------------

class InventoryIn(ApiModel):
    product_name: str = Field(min_length=1, max_length=80)
    variety: str = Field(min_length=1, max_length=80)
    quantity: Decimal = Field(ge=0, le=10_000_000, max_digits=12, decimal_places=2)
    price: Decimal = Field(ge=0, le=100_000_000, max_digits=12, decimal_places=2)
    harvest_date: date | None = None
    quality_grade: str = Field(default="", max_length=20)
    packaging_type: str = Field(default="", max_length=30)

    _c = field_validator("product_name", "variety", "quality_grade", "packaging_type")(lambda cls, v: _clean(v))


class InventoryOut(ApiModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
    id: str
    product_name: str
    variety: str
    quantity: float
    price: float
    harvest_date: str | None
    quality_grade: str
    packaging_type: str
    updated_at: str


# --- Forum ------------------------------------------------------------------

class PostIn(ApiModel):
    text: str = Field(min_length=1, max_length=2000)
    _c = field_validator("text")(lambda cls, v: _clean(v))


class ReplyOut(ApiModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
    id: str
    text: str
    author_id: str
    author_name: str
    created_at: str


class PostOut(ApiModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
    id: str
    text: str
    author_id: str
    author_name: str
    likes: int
    liked_by_me: bool = False
    reply_count: int
    created_at: str
    edited_at: str | None = None
    replies: list[ReplyOut] = []


class PostPage(ApiModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
    items: list[PostOut]
    next_cursor: str | None = None


# --- Crop issue reports -----------------------------------------------------

REPORT_CATEGORIES = Literal[
    "pest", "disease", "economic", "tools", "local_people", "weather", "soil_climate",
    "biohazard", "transport", "bribery", "natural_calamity", "wildfire", "forest", "other",
]


class ReportOut(ApiModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
    id: str
    owner_id: str
    crop: str
    category: str
    severity: str
    description: str
    district: str | None
    status: str
    photo_urls: list[str] = []
    admin_note: str | None = None
    owner_name: str | None = None   # admin views only
    owner_phone: str | None = None  # admin views only, so an officer can call the farmer
    created_at: str
    updated_at: str


class ReportStatusUpdate(ApiModel):
    status: Literal["open", "in_review", "resolved", "rejected"]
    admin_note: str | None = Field(default=None, max_length=1000)


# --- AI ---------------------------------------------------------------------

class AskRequest(ApiModel):
    question: str = Field(min_length=2, max_length=1000)
    language: Literal["en", "hi"] = "en"


class Source(ApiModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
    title: str
    document: str
    score: float


class AskResponse(ApiModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
    answer: str
    sources: list[Source]
    grounded: bool


class Prediction(ApiModel):
    label: str
    probability: float


class VisionResponse(ApiModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
    predictions: list[Prediction]
    model_version: str
