"""Request dependencies: authentication, authorisation and rate limits."""
from dataclasses import dataclass

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.middleware import client_ip, limiter
from app.core.security import decode_access_token

_bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class CurrentUser:
    id: str
    role: str

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


def get_current_user(creds: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> CurrentUser:
    if creds is None or creds.scheme.lower() != "bearer":
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not authenticated",
                            headers={"WWW-Authenticate": "Bearer"})
    try:
        claims = decode_access_token(creds.credentials)
    except jwt.PyJWTError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token",
                            headers={"WWW-Authenticate": "Bearer"})
    return CurrentUser(id=claims["sub"], role=claims.get("role", "user"))


def require_admin(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    if not user.is_admin:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admin access required")
    return user


def rate_limit(name: str, limit: int, window_s: int):
    """Limit by client IP (and by user when authenticated)."""
    def _dep(request: Request):
        key = f"{name}:{client_ip(request)}"
        if not limiter.hit(key, limit, window_s):
            raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Too many requests, please slow down.",
                                headers={"Retry-After": str(window_s)})
    return _dep


def require_csrf_header(request: Request):
    """Cookie-authenticated endpoints need a custom header.

    Browsers cannot add custom headers to cross-site requests without a CORS
    preflight, and CORS only allows our own origins, so this blocks CSRF on
    the refresh-token cookie (in addition to SameSite=Strict).
    """
    if request.headers.get("x-requested-with") != "ameer-kisaan":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Missing CSRF header")
