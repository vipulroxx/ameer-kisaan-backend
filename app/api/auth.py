import re

from boto3.dynamodb.conditions import Key
from fastapi import APIRouter, BackgroundTasks, Cookie, Depends, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse

from app.core import audit, security
from app.core.config import get_settings
from app.core.deps import CurrentUser, get_current_user, rate_limit, require_admin, require_csrf_header
from app.core.middleware import client_ip
from app.db.dynamo import table
from app.schemas import (
    AdminUserUpdate,
    LoginRequest,
    PasswordChange,
    ProfileUpdate,
    RegisterRequest,
    TokenResponse,
    UserOut,
)
from app.services import users

router = APIRouter()
REFRESH_COOKIE = "ak_refresh"


def _set_refresh_cookie(response: Response, token: str) -> None:
    s = get_settings()
    response.set_cookie(
        REFRESH_COOKIE, token, max_age=s.refresh_token_ttl_days * 86400, httponly=True,
        secure=s.cookie_secure, samesite="strict", path=f"{s.api_prefix}/auth",
    )


def _clear_refresh_cookie(response: Response) -> None:
    s = get_settings()
    response.delete_cookie(REFRESH_COOKIE, path=f"{s.api_prefix}/auth", secure=s.cookie_secure,
                           httponly=True, samesite="strict")


def _token_response(user: dict) -> TokenResponse:
    access, ttl = security.create_access_token(user["id"], user.get("role", "user"))
    return TokenResponse(access_token=access, expires_in=ttl, user=UserOut(**users.to_public(user)))


@router.post("/auth/register", response_model=UserOut, status_code=201,
             dependencies=[Depends(rate_limit("register", 5, 600))])
def register(body: RegisterRequest, request: Request, bg: BackgroundTasks):
    problems = security.validate_password_strength(body.password, str(body.email))
    if problems:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, " ".join(problems))
    try:
        user = users.create_user(body.model_dump())
    except users.EmailTaken:
        raise HTTPException(status.HTTP_409_CONFLICT, "An account with this email already exists.")
    bg.add_task(audit.record, "user.register", user["id"], ip=client_ip(request))
    return UserOut(**users.to_public(user))


@router.post("/auth/login", response_model=TokenResponse,
             dependencies=[Depends(rate_limit("login", 10, 60))])
def login(body: LoginRequest, request: Request, response: Response, bg: BackgroundTasks):
    ip = client_ip(request)
    try:
        user = users.authenticate(str(body.email), body.password)
    except users.AccountLocked as exc:
        bg.add_task(audit.record, "auth.login", None, ip=ip, outcome="locked",
                    detail={"email": security.mask_email(str(body.email))})
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, str(exc))
    except users.AuthError as exc:
        bg.add_task(audit.record, "auth.login", None, ip=ip, outcome="failure",
                    detail={"email": security.mask_email(str(body.email))})
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(exc))
    _set_refresh_cookie(response, users.start_session(user["id"]))
    bg.add_task(audit.record, "auth.login", user["id"], ip=ip)
    return _token_response(user)


@router.post("/auth/refresh", response_model=TokenResponse,
             dependencies=[Depends(require_csrf_header), Depends(rate_limit("refresh", 30, 60))])
def refresh(request: Request, response: Response, bg: BackgroundTasks,
            ak_refresh: str | None = Cookie(default=None)):
    if not ak_refresh:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not logged in.")
    try:
        user, new_token = users.rotate_session(ak_refresh)
    except users.AuthError as exc:
        bg.add_task(audit.record, "auth.refresh", None, ip=client_ip(request), outcome="failure",
                    detail={"reason": str(exc)})
        failed = JSONResponse({"detail": str(exc)}, status_code=401, background=bg)
        _clear_refresh_cookie(failed)
        return failed
    _set_refresh_cookie(response, new_token)
    return _token_response(user)


@router.post("/auth/logout", status_code=204, dependencies=[Depends(require_csrf_header)])
def logout(request: Request, response: Response, bg: BackgroundTasks,
           ak_refresh: str | None = Cookie(default=None)):
    if ak_refresh:
        uid = users.end_session(ak_refresh)
        bg.add_task(audit.record, "auth.logout", uid, ip=client_ip(request))
    _clear_refresh_cookie(response)
    response.status_code = 204
    return response


# --- Current user -----------------------------------------------------------

@router.get("/users/me", response_model=UserOut)
def me(user: CurrentUser = Depends(get_current_user)):
    record = users.get_user(user.id)
    if not record or record.get("disabled"):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Account unavailable")
    return UserOut(**users.to_public(record))


@router.patch("/users/me", response_model=UserOut)
def update_me(body: ProfileUpdate, request: Request, bg: BackgroundTasks,
              user: CurrentUser = Depends(get_current_user)):
    updated = users.update_profile(user.id, body.model_dump(exclude_unset=True))
    bg.add_task(audit.record, "user.update_profile", user.id, ip=client_ip(request),
                detail={"fields": ",".join(body.model_dump(exclude_unset=True))})
    return UserOut(**users.to_public(updated))


@router.post("/users/me/password", response_model=TokenResponse,
             dependencies=[Depends(rate_limit("password", 5, 600))])
def change_password(body: PasswordChange, request: Request, response: Response, bg: BackgroundTasks,
                    user: CurrentUser = Depends(get_current_user)):
    record = users.get_user(user.id)
    if not record or not security.verify_password(body.current_password, record["password_hash"]):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Current password is incorrect.")
    problems = security.validate_password_strength(body.new_password, record["email"])
    if problems:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, " ".join(problems))
    users.set_password(user.id, body.new_password)   # signs out all existing sessions
    _set_refresh_cookie(response, users.start_session(user.id))  # ...except this fresh one
    bg.add_task(audit.record, "user.change_password", user.id, ip=client_ip(request))
    return _token_response(users.get_user(user.id))


# --- Admin ------------------------------------------------------------------

@router.get("/admin/users")
def admin_list_users(limit: int = 50, cursor: str | None = None, admin: CurrentUser = Depends(require_admin)):
    items, nxt = users.list_users(min(max(limit, 1), 100), cursor)
    return {"items": [UserOut(**users.to_public(u)).model_dump(by_alias=True) for u in items], "nextCursor": nxt}


@router.patch("/admin/users/{user_id}", response_model=UserOut)
def admin_update_user(user_id: str, body: AdminUserUpdate, request: Request, bg: BackgroundTasks,
                      admin: CurrentUser = Depends(require_admin)):
    if user_id == admin.id and body.role == "user":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "You cannot remove your own admin role.")
    if not users.get_user(user_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found")
    changes = body.model_dump(exclude_unset=True)
    updated = users.update_profile(user_id, changes)
    if changes.get("disabled") or "role" in changes:
        users.set_sessions_cutoff(user_id)
    bg.add_task(audit.record, "admin.update_user", admin.id, target=user_id, ip=client_ip(request),
                detail=changes)
    return UserOut(**users.to_public(updated))


@router.get("/admin/audit")
def admin_audit(day: str, admin: CurrentUser = Depends(require_admin)):
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", day):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "day must be YYYY-MM-DD")
    resp = table("audit").query(KeyConditionExpression=Key("day").eq(day), ScanIndexForward=False, Limit=200)
    return {"items": resp.get("Items", [])}
