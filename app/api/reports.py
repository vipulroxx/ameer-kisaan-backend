"""Crop issue reports with photo evidence.

Photos are validated and re-encoded server-side (EXIF/GPS stripped), stored in
a private, encrypted S3 bucket, and only ever exposed through short-lived
presigned URLs generated for the report's owner or an admin.
"""
import uuid
from datetime import datetime, timezone

from boto3.dynamodb.conditions import Key
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile

from app.core import audit
from app.core.config import get_settings
from app.core.deps import CurrentUser, get_current_user, rate_limit, require_admin
from app.db import storage
from app.db.dynamo import table
from app.schemas import REPORT_CATEGORIES, ReportOut, ReportStatusUpdate
from app.services import users

router = APIRouter(prefix="/reports", tags=["reports"])
MAX_PHOTOS = 4
SEVERITIES = {"low", "medium", "high", "critical"}


def _out(r: dict) -> ReportOut:
    return ReportOut(
        id=r["report_id"], owner_id=r["owner_id"], crop=r["crop"], category=r["category"], severity=r["severity"],
        description=r["description"], district=r.get("district"), status=r["status"],
        photo_urls=[storage.presigned_get(k) for k in r.get("photo_keys", [])], admin_note=r.get("admin_note"),
        created_at=r["created_at"], updated_at=r["updated_at"],
    )


async def read_limited(upload: UploadFile) -> bytes:
    limit = get_settings().max_upload_mb * 1024 * 1024
    data = await upload.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"Each photo must be under {get_settings().max_upload_mb} MB.")
    return data


@router.post("", response_model=ReportOut, status_code=201,
             dependencies=[Depends(rate_limit("report", 10, 600))])
async def create_report(
    bg: BackgroundTasks,
    crop: str = Form(min_length=1, max_length=60),
    category: REPORT_CATEGORIES = Form(),
    severity: str = Form("medium"),
    description: str = Form(min_length=5, max_length=2000),
    district: str | None = Form(None, max_length=60),
    photos: list[UploadFile] = File(default_factory=list),
    user: CurrentUser = Depends(get_current_user),
):
    if severity not in SEVERITIES:
        raise HTTPException(422, f"severity must be one of {sorted(SEVERITIES)}")
    if len(photos) > MAX_PHOTOS:
        raise HTTPException(422, f"At most {MAX_PHOTOS} photos per report.")
    report_id = uuid.uuid4().hex
    keys = []
    for p in photos:
        try:
            clean, ctype = storage.sanitize_image(await read_limited(p))
        except storage.InvalidImage as exc:
            for k in keys:
                storage.delete_object(k)
            raise HTTPException(415, str(exc))
        keys.append(storage.put_image(f"reports/{user.id}/{report_id}", clean, ctype))
    now = datetime.now(timezone.utc).isoformat()
    item = {"owner_id": user.id, "report_id": report_id, "crop": crop.strip(), "category": category,
            "severity": severity, "description": description.strip(), "status": "open", "photo_keys": keys,
            "created_at": now, "updated_at": now}
    if district:
        item["district"] = district.strip()
    table("reports").put_item(Item=item)
    bg.add_task(audit.record, "report.create", user.id, target=report_id, detail={"photos": len(keys)})
    return _out(item)


@router.get("", response_model=list[ReportOut])
def my_reports(user: CurrentUser = Depends(get_current_user)):
    resp = table("reports").query(KeyConditionExpression=Key("owner_id").eq(user.id))
    return [_out(r) for r in sorted(resp.get("Items", []), key=lambda r: r["created_at"], reverse=True)]


@router.get("/admin", response_model=list[ReportOut])
def reports_by_status(status: str = "open", admin: CurrentUser = Depends(require_admin)):
    resp = table("reports").query(IndexName="status-index", KeyConditionExpression=Key("status").eq(status),
                                  ScanIndexForward=False, Limit=100)
    owners: dict = {}
    return [_with_owner(_out(r), owners) for r in resp.get("Items", [])]


def _with_owner(rep: ReportOut, owners: dict | None = None) -> ReportOut:
    """Admin views: who reported it and how to reach them (one profile lookup per farmer)."""
    owners = {} if owners is None else owners
    if rep.owner_id not in owners:
        owners[rep.owner_id] = users.get_user(rep.owner_id)
    u = owners[rep.owner_id]
    return rep.model_copy(update={"owner_name": u["username"] if u else "Former member",
                                  "owner_phone": u.get("phone_number") if u else None})


@router.patch("/{owner_id}/{report_id}", response_model=ReportOut)
def update_status(owner_id: str, report_id: str, body: ReportStatusUpdate, bg: BackgroundTasks,
                  admin: CurrentUser = Depends(require_admin)):
    key = {"owner_id": owner_id, "report_id": report_id}
    if "Item" not in table("reports").get_item(Key=key):
        raise HTTPException(404, "Report not found")
    updated = table("reports").update_item(
        Key=key, UpdateExpression="SET #s = :s, admin_note = :n, updated_at = :u",
        ExpressionAttributeNames={"#s": "status"},
        ExpressionAttributeValues={":s": body.status, ":n": body.admin_note,
                                   ":u": datetime.now(timezone.utc).isoformat()},
        ReturnValues="ALL_NEW")["Attributes"]
    bg.add_task(audit.record, "report.status", admin.id, target=report_id, detail={"status": body.status})
    return _with_owner(_out(updated))
