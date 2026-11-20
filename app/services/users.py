"""User accounts, login lockout and refresh-token sessions."""
import time
import uuid
from datetime import datetime, timezone

from boto3.dynamodb.conditions import Attr
from botocore.exceptions import ClientError

from app.core import security
from app.core.config import get_settings
from app.db.dynamo import dynamodb, table


class AuthError(Exception):
    """Raised for any authentication failure. The message is safe to show users."""


class AccountLocked(AuthError):
    pass


class EmailTaken(Exception):
    pass


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_ms() -> int:
    """Session cut-offs use milliseconds so a re-login right after a cut-off is not revoked by it."""
    time.sleep(0.002)  # guarantee strictly increasing values within one process
    return int(time.time() * 1000)


def normalize_email(email: str) -> str:
    return email.strip().lower()


# --- Users ------------------------------------------------------------------

def create_user(data: dict) -> dict:
    s = get_settings()
    cipher = security.get_cipher()
    email = normalize_email(data["email"])
    user_id = uuid.uuid4().hex
    item = {
        "pk": f"USER#{user_id}",
        "id": user_id,
        "email": email,
        "username": data["username"],
        "phone_number": data["phone_number"],
        "user_type": data.get("user_type", "Farmer"),
        "role": "user",  # never taken from the request: admins are promoted explicitly
        "preferred_crops": data.get("preferred_crops") or [],
        "farm_id": data.get("farm_id"),
        "password_hash": security.hash_password(data["password"]),
        "sessions_valid_after": 0,
        "disabled": False,
        "created_at": _now_iso(),
    }
    if data.get("aadhar_id"):
        item["aadhar_enc"] = cipher.encrypt(data["aadhar_id"], f"aadhaar:{user_id}")
        item["aadhar_masked"] = security.mask_aadhaar(data["aadhar_id"])
    if data.get("pan_card_number"):
        item["pan_enc"] = cipher.encrypt(data["pan_card_number"], f"pan:{user_id}")
        item["pan_masked"] = security.mask_pan(data["pan_card_number"])
    item = {k: v for k, v in item.items() if v is not None}

    # The resource's client serialises plain Python values for us.
    client = dynamodb().meta.client
    users = s.table("users")
    try:
        # Both writes succeed or neither does: the EMAIL# lock guarantees uniqueness.
        client.transact_write_items(TransactItems=[
            {"Put": {"TableName": users, "Item": item, "ConditionExpression": "attribute_not_exists(pk)"}},
            {"Put": {"TableName": users, "Item": {"pk": f"EMAIL#{email}", "user_id": user_id},
                     "ConditionExpression": "attribute_not_exists(pk)"}},
        ])
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "TransactionCanceledException":
            raise EmailTaken() from exc
        raise
    return item


def get_user(user_id: str) -> dict | None:
    return table("users").get_item(Key={"pk": f"USER#{user_id}"}, ConsistentRead=True).get("Item")


def get_user_by_email(email: str) -> dict | None:
    lock = table("users").get_item(Key={"pk": f"EMAIL#{normalize_email(email)}"}, ConsistentRead=True).get("Item")
    return get_user(lock["user_id"]) if lock else None


def to_public(user: dict) -> dict:
    return {
        "id": user["id"], "username": user["username"], "email": user["email"],
        "phone_number": user["phone_number"], "user_type": user.get("user_type", "Farmer"),
        "role": user.get("role", "user"), "preferred_crops": list(user.get("preferred_crops", [])),
        "farm_id": user.get("farm_id"), "aadhar_id": user.get("aadhar_masked"),
        "pan_card_number": user.get("pan_masked"), "created_at": user["created_at"],
    }


def display_name(user_id: str) -> str:
    u = get_user(user_id)
    return u["username"] if u else "Former member"


def update_profile(user_id: str, changes: dict) -> dict:
    changes = {k: v for k, v in changes.items() if v is not None}
    if not changes:
        return get_user(user_id)
    names = {f"#{k}": k for k in changes}
    values = {f":{k}": v for k, v in changes.items()}
    expr = "SET " + ", ".join(f"#{k} = :{k}" for k in changes)
    return table("users").update_item(
        Key={"pk": f"USER#{user_id}"}, UpdateExpression=expr, ExpressionAttributeNames=names,
        ExpressionAttributeValues=values, ConditionExpression="attribute_exists(pk)", ReturnValues="ALL_NEW",
    )["Attributes"]


def set_password(user_id: str, new_password: str) -> None:
    # Bumping sessions_valid_after signs out every other device.
    table("users").update_item(
        Key={"pk": f"USER#{user_id}"},
        UpdateExpression="SET password_hash = :h, sessions_valid_after = :t",
        ExpressionAttributeValues={":h": security.hash_password(new_password), ":t": _now_ms()},
    )


def set_sessions_cutoff(user_id: str) -> None:
    """Invalidate every refresh session issued before now (used on role change / disable)."""
    table("users").update_item(Key={"pk": f"USER#{user_id}"}, UpdateExpression="SET sessions_valid_after = :t",
                               ExpressionAttributeValues={":t": _now_ms()})


def list_users(limit: int = 50, cursor: str | None = None) -> tuple[list[dict], str | None]:
    kwargs = {"FilterExpression": Attr("pk").begins_with("USER#"), "Limit": limit}
    if cursor:
        kwargs["ExclusiveStartKey"] = {"pk": cursor}
    resp = table("users").scan(**kwargs)
    nxt = resp.get("LastEvaluatedKey", {}).get("pk")
    return resp.get("Items", []), nxt


# --- Login with lockout -----------------------------------------------------

def authenticate(email: str, password: str) -> dict:
    s = get_settings()
    email = normalize_email(email)
    sessions = table("sessions")
    fail_key = {"pk": f"LOGINFAIL#{email}"}
    now = int(time.time())

    fails = sessions.get_item(Key=fail_key, ConsistentRead=True).get("Item")
    if fails and int(fails.get("locked_until", 0)) > now:
        raise AccountLocked("Too many failed attempts. Try again later.")

    user = get_user_by_email(email)
    ok = security.verify_password(password, user["password_hash"] if user else None)
    if not ok or user.get("disabled"):
        count = sessions.update_item(
            Key=fail_key,
            UpdateExpression="ADD attempts :one SET expires_at = :exp",
            ExpressionAttributeValues={":one": 1, ":exp": now + s.login_lockout_minutes * 60},
            ReturnValues="UPDATED_NEW",
        )["Attributes"]["attempts"]
        if count >= s.login_max_attempts:
            sessions.update_item(Key=fail_key, UpdateExpression="SET locked_until = :u, attempts = :z",
                                 ExpressionAttributeValues={":u": now + s.login_lockout_minutes * 60, ":z": 0})
        raise AuthError("Invalid email or password.")

    sessions.delete_item(Key=fail_key)
    if security.needs_rehash(user["password_hash"]):
        table("users").update_item(Key={"pk": user["pk"]}, UpdateExpression="SET password_hash = :h",
                                   ExpressionAttributeValues={":h": security.hash_password(password)})
    return user


# --- Refresh-token sessions with rotation and reuse detection ---------------

def start_session(user_id: str) -> str:
    family = uuid.uuid4().hex
    exp = int(time.time()) + get_settings().refresh_token_ttl_days * 86400
    table("sessions").put_item(Item={"pk": f"FAMILY#{family}", "user_id": user_id, "revoked": False,
                                     "created_at": _now_ms(), "expires_at": exp})
    return _issue_refresh(user_id, family, exp)


def _issue_refresh(user_id: str, family: str, family_exp: int) -> str:
    token = security.new_refresh_token()
    exp = min(int(time.time()) + get_settings().refresh_token_ttl_days * 86400, family_exp)
    table("sessions").put_item(Item={"pk": f"REFRESH#{security.hash_token(token)}", "user_id": user_id,
                                     "family": family, "used": False, "expires_at": exp})
    return token


def rotate_session(token: str) -> tuple[dict, str]:
    """Exchange a refresh token for a new one. Each token works exactly once.

    If an already-used token is presented, someone has a stolen copy: the whole
    session family is revoked, signing out both the attacker and the victim.
    """
    sessions = table("sessions")
    now = int(time.time())
    rec = sessions.get_item(Key={"pk": f"REFRESH#{security.hash_token(token)}"}, ConsistentRead=True).get("Item")
    if not rec or int(rec["expires_at"]) < now:
        raise AuthError("Session expired. Please log in again.")
    fam = sessions.get_item(Key={"pk": f"FAMILY#{rec['family']}"}, ConsistentRead=True).get("Item")
    if not fam or fam.get("revoked"):
        raise AuthError("Session expired. Please log in again.")
    if rec.get("used"):
        revoke_family(rec["family"])
        raise AuthError("Session reuse detected. Please log in again.")
    try:
        sessions.update_item(Key={"pk": rec["pk"]}, UpdateExpression="SET used = :t",
                             ConditionExpression="used = :f", ExpressionAttributeValues={":t": True, ":f": False})
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "ConditionalCheckFailedException":
            revoke_family(rec["family"])
            raise AuthError("Session reuse detected. Please log in again.") from exc
        raise

    user = get_user(rec["user_id"])
    if not user or user.get("disabled") or int(fam["created_at"]) < int(user.get("sessions_valid_after", 0)):
        revoke_family(rec["family"])
        raise AuthError("Session expired. Please log in again.")
    return user, _issue_refresh(user["id"], rec["family"], int(fam["expires_at"]))


def revoke_family(family: str) -> None:
    table("sessions").update_item(Key={"pk": f"FAMILY#{family}"}, UpdateExpression="SET revoked = :t",
                                  ExpressionAttributeValues={":t": True})


def end_session(token: str) -> str | None:
    rec = table("sessions").get_item(Key={"pk": f"REFRESH#{security.hash_token(token)}"}).get("Item")
    if rec:
        revoke_family(rec["family"])
        return rec["user_id"]
    return None
