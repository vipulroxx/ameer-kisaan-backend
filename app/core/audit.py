"""Append-only audit trail.

Writes happen in a FastAPI background task, after the response is sent, so
audit logging never adds latency to the user's request. Failures are logged
and never break the request. Events never contain passwords, tokens or
plaintext identifiers.
"""
import logging
import uuid
from datetime import datetime, timezone

from app.db.dynamo import table

log = logging.getLogger("audit")


def record(action: str, actor_id: str | None, *, target: str | None = None,
           ip: str | None = None, outcome: str = "success", detail: dict | None = None) -> None:
    now = datetime.now(timezone.utc)
    item = {
        "day": now.strftime("%Y-%m-%d"),
        "event_id": f"{now.isoformat()}#{uuid.uuid4().hex[:12]}",
        "action": action,
        "actor_id": actor_id or "anonymous",
        "outcome": outcome,
        "at": now.isoformat(),
    }
    if target:
        item["target"] = target
    if ip:
        item["ip"] = ip
    if detail:
        item["detail"] = {k: str(v)[:200] for k, v in detail.items()}
    try:
        # attribute_not_exists makes every event write-once
        table("audit").put_item(Item=item, ConditionExpression="attribute_not_exists(event_id)")
    except Exception:  # pragma: no cover - logged, never raised
        log.exception("audit write failed", extra={"action": action})
