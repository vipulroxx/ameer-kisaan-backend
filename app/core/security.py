"""Password hashing, token issuing and field-level encryption."""
import base64
import hashlib
import hmac
import os
import re
import secrets
import uuid
from datetime import datetime, timedelta, timezone

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from app.core.config import get_settings

# Argon2id with the library's OWASP-aligned defaults (memory-hard, salted).
_hasher = PasswordHasher()
# Used so a login for an unknown email costs the same time as a real one,
# which stops attackers discovering registered emails by timing responses.
_DUMMY_HASH = _hasher.hash(secrets.token_urlsafe(16))

COMMON_PASSWORDS = {
    "password", "password1", "password123", "1234567890", "qwerty12345", "iloveyou12",
    "admin12345", "welcome123", "kisaan12345", "farmer12345", "letmein1234",
}


def validate_password_strength(password: str, email: str | None = None) -> list[str]:
    problems = []
    if len(password) < 10:
        problems.append("Password must be at least 10 characters.")
    if len(password) > 128:
        problems.append("Password must be at most 128 characters.")
    if not re.search(r"[A-Za-z]", password) or not re.search(r"\d", password):
        problems.append("Password must contain both letters and numbers.")
    if password.lower() in COMMON_PASSWORDS:
        problems.append("Password is too common.")
    if email and email.split("@")[0].lower() in password.lower():
        problems.append("Password must not contain your email name.")
    return problems


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, password_hash: str | None) -> bool:
    try:
        return _hasher.verify(password_hash or _DUMMY_HASH, password) and password_hash is not None
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def needs_rehash(password_hash: str) -> bool:
    return _hasher.check_needs_rehash(password_hash)


# --- Access tokens (short-lived JWT) -----------------------------------------

def create_access_token(user_id: str, role: str) -> tuple[str, int]:
    s = get_settings()
    now = datetime.now(timezone.utc)
    ttl = s.access_token_ttl_minutes * 60
    payload = {
        "sub": user_id,
        "role": role,
        "type": "access",
        "iss": s.jwt_issuer,
        "aud": s.jwt_audience,
        "iat": now,
        "nbf": now,
        "exp": now + timedelta(seconds=ttl),
        "jti": uuid.uuid4().hex,
    }
    return jwt.encode(payload, s.jwt_secret, algorithm="HS256"), ttl


def decode_access_token(token: str) -> dict:
    s = get_settings()
    claims = jwt.decode(
        token,
        s.jwt_secret,
        algorithms=["HS256"],  # pinned: never trust the token's own "alg" header
        audience=s.jwt_audience,
        issuer=s.jwt_issuer,
        options={"require": ["exp", "iat", "sub", "aud", "iss"]},
    )
    if claims.get("type") != "access":
        raise jwt.InvalidTokenError("wrong token type")
    return claims


# --- Refresh tokens (opaque, stored hashed, rotated on every use) -------------

def new_refresh_token() -> str:
    return secrets.token_urlsafe(48)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


# --- Field encryption for sensitive identifiers -------------------------------

class FieldCipher:
    """AES-256-GCM authenticated encryption for individual sensitive fields."""

    def __init__(self, key_b64: str):
        key = base64.b64decode(key_b64)
        if len(key) != 32:
            raise ValueError("FIELD_ENCRYPTION_KEY must decode to exactly 32 bytes")
        self._aead = AESGCM(key)
        self._mac_key = hashlib.sha256(b"ak-blind-index" + key).digest()

    def encrypt(self, plaintext: str, context: str) -> str:
        nonce = os.urandom(12)
        ct = self._aead.encrypt(nonce, plaintext.encode(), context.encode())
        return "v1:" + base64.b64encode(nonce + ct).decode()

    def decrypt(self, token: str, context: str) -> str:
        raw = base64.b64decode(token.removeprefix("v1:"))
        return self._aead.decrypt(raw[:12], raw[12:], context.encode()).decode()

    def blind_index(self, value: str) -> str:
        """Deterministic keyed hash, so duplicates can be detected without storing plaintext."""
        return hmac.new(self._mac_key, value.encode(), hashlib.sha256).hexdigest()


_cipher: FieldCipher | None = None


def get_cipher() -> FieldCipher:
    global _cipher
    if _cipher is None:
        _cipher = FieldCipher(get_settings().field_encryption_key)
    return _cipher


def mask_aadhaar(value: str) -> str:
    digits = re.sub(r"\D", "", value)
    return f"XXXX-XXXX-{digits[-4:]}" if len(digits) >= 4 else "XXXX"


def mask_pan(value: str) -> str:
    return f"{value[:2]}XXXXXX{value[-2:]}" if len(value) == 10 else "XXXXXXXXXX"


def mask_email(email: str) -> str:
    name, _, domain = email.partition("@")
    return f"{name[:1]}***@{domain}"
