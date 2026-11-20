"""Signed photo links (PHOTO_URL_MODE=proxy).

Report photos stay in a private bucket. The API hands out links that expire
after PRESIGNED_URL_TTL_SECONDS and are HMAC-signed for one object key, the
same guarantees as an S3 presigned URL, but served by us so any S3-compatible
store works and the browser only ever talks to our own origin.
"""
import re

from botocore.exceptions import ClientError
from fastapi import APIRouter, HTTPException, Query, Response

from app.db import storage

router = APIRouter(tags=["photos"])
# Keys are always reports/<user id>/<report id>/<uuid>.jpg (see storage.put_image).
KEY_RE = re.compile(r"^reports/[A-Za-z0-9_-]{1,64}/[A-Za-z0-9_-]{1,64}/[0-9a-f]{32}\.jpg$")


@router.get("/photos/{key:path}", include_in_schema=False)
def photo(key: str, exp: int = Query(...), sig: str = Query(..., max_length=128)):
    if not KEY_RE.match(key) or not storage.verify_photo_link(key, exp, sig):
        raise HTTPException(403, "This photo link is invalid or has expired.")
    try:
        data, ctype = storage.get_object(key)
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("NoSuchKey", "404"):
            raise HTTPException(404, "Photo not found")
        raise
    return Response(data, media_type=ctype, headers={"Cache-Control": "private, max-age=300"})
