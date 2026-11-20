"""S3 object storage (MinIO locally) with server-side image sanitising."""
import hashlib
import hmac
import io
import time
import uuid
from functools import lru_cache
from urllib.parse import urlencode

import boto3
from botocore.config import Config
from PIL import Image, UnidentifiedImageError

from app.core.config import get_settings

ALLOWED_FORMATS = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
MAX_PIXELS = 40_000_000  # refuse decompression bombs
Image.MAX_IMAGE_PIXELS = MAX_PIXELS


class InvalidImage(ValueError):
    pass


def sanitize_image(data: bytes, max_side: int = 2048) -> tuple[bytes, str]:
    """Decode, validate and re-encode an uploaded image.

    The file's own extension and Content-Type are ignored: we trust only what
    Pillow can actually decode. Re-encoding strips EXIF metadata (including GPS
    coordinates of the farmer's field) and any payload hidden after the image.
    """
    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt = probe.format
            probe.verify()
        if fmt not in ALLOWED_FORMATS:
            raise InvalidImage("Only JPEG, PNG or WEBP images are allowed.")
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise InvalidImage("File is not a valid image.") from exc

    img = img.convert("RGB")
    img.thumbnail((max_side, max_side))
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=88, optimize=True)  # no exif= argument -> metadata dropped
    return out.getvalue(), "image/jpeg"


@lru_cache
def _s3(public: bool = False):
    s = get_settings()
    endpoint = (s.s3_public_endpoint_url or s.s3_endpoint_url) if public else s.s3_endpoint_url
    return boto3.client(
        "s3",
        region_name=s.aws_region,
        endpoint_url=endpoint,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path" if endpoint else "auto"}),
    )


def put_image(prefix: str, data: bytes, content_type: str) -> str:
    key = f"{prefix}/{uuid.uuid4().hex}.jpg"
    _s3().put_object(
        Bucket=get_settings().s3_bucket, Key=key, Body=data, ContentType=content_type,
    )  # encrypted by the bucket's default KMS key (see infra/terraform/data.tf)
    return key


def presigned_get(key: str) -> str:
    """A short-lived URL the browser can load the (private) photo from."""
    s = get_settings()
    if s.photo_url_mode == "proxy":
        exp = int(time.time()) + s.presigned_url_ttl_seconds
        return f"{s.api_prefix}/photos/{key}?" + urlencode({"exp": exp, "sig": photo_signature(key, exp)})
    return _s3(public=True).generate_presigned_url(
        "get_object", Params={"Bucket": s.s3_bucket, "Key": key}, ExpiresIn=s.presigned_url_ttl_seconds
    )


def _photo_key() -> bytes:
    # Derived, purpose-bound key: a photo link can never be replayed as a JWT or vice versa.
    return hmac.new(get_settings().jwt_secret.encode(), b"ameer-kisaan:photo-url:v1", hashlib.sha256).digest()


def photo_signature(key: str, exp: int) -> str:
    return hmac.new(_photo_key(), f"{key}\n{exp}".encode(), hashlib.sha256).hexdigest()


def verify_photo_link(key: str, exp: int, sig: str) -> bool:
    return exp >= int(time.time()) and hmac.compare_digest(photo_signature(key, exp), sig)


def get_object(key: str) -> tuple[bytes, str]:
    obj = _s3().get_object(Bucket=get_settings().s3_bucket, Key=key)
    return obj["Body"].read(), obj.get("ContentType") or "image/jpeg"


def delete_object(key: str) -> None:
    _s3().delete_object(Bucket=get_settings().s3_bucket, Key=key)


def ensure_bucket() -> None:
    """Local/dev helper. In AWS the bucket is created (private, encrypted) by Terraform."""
    s3, bucket = _s3(), get_settings().s3_bucket
    try:
        s3.head_bucket(Bucket=bucket)
        return
    except Exception:  # noqa: S110 - bucket missing; create it below
        pass
    region = get_settings().aws_region
    try:
        s3.create_bucket(Bucket=bucket, CreateBucketConfiguration={"LocationConstraint": region})
    except Exception:
        s3.create_bucket(Bucket=bucket)  # some local S3 servers reject a location constraint
