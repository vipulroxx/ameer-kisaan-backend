"""DynamoDB access and table definitions.

The same table definitions are used by:
  * scripts/bootstrap.py to create tables in DynamoDB Local,
  * the test suite (moto), and
  * infra/terraform (kept in sync by hand, see dynamodb.tf).
"""
import logging
from functools import lru_cache

import boto3
from botocore.config import Config

from app.core.config import get_settings

# name -> (key schema, attribute definitions, GSIs, ttl attribute)
TABLES: dict[str, dict] = {
    # USER#<id> profile items and EMAIL#<email> uniqueness locks
    "users": {"keys": [("pk", "HASH")], "attrs": {"pk": "S"}},
    # REFRESH#<hash>, FAMILY#<id>, LOGINFAIL#<email>
    "sessions": {"keys": [("pk", "HASH")], "attrs": {"pk": "S"}, "ttl": "expires_at"},
    "inventory": {"keys": [("owner_id", "HASH"), ("item_id", "RANGE")],
                  "attrs": {"owner_id": "S", "item_id": "S"}},
    # POST#<id> / META, POST#<id> / REPLY#<ts>#<id>, POST#<id> / LIKE#<user>
    "forum": {
        "keys": [("pk", "HASH"), ("sk", "RANGE")],
        "attrs": {"pk": "S", "sk": "S", "feed": "S", "created_at": "S"},
        "gsis": [{"name": "feed-index", "keys": [("feed", "HASH"), ("created_at", "RANGE")]}],
    },
    "reports": {
        "keys": [("owner_id", "HASH"), ("report_id", "RANGE")],
        "attrs": {"owner_id": "S", "report_id": "S", "status": "S", "created_at": "S"},
        "gsis": [{"name": "status-index", "keys": [("status", "HASH"), ("created_at", "RANGE")]}],
    },
    # Append-only. In AWS the app's IAM role may only PutItem/Query here.
    "audit": {"keys": [("day", "HASH"), ("event_id", "RANGE")], "attrs": {"day": "S", "event_id": "S"}},
    "chat": {"keys": [("user_id", "HASH"), ("message_id", "RANGE")],
             "attrs": {"user_id": "S", "message_id": "S"}, "ttl": "expires_at"},
}


@lru_cache
def dynamodb():
    """boto3 DynamoDB resource, or the Postgres-backed equivalent when DATABASE_URL is set."""
    s = get_settings()
    if s.database_url:
        from app.db import pgtable
        return pgtable.PgResource(pgtable.backend(s.database_url, s.database_pool_size), TABLES, s.table_prefix)
    return boto3.resource(
        "dynamodb",
        region_name=s.aws_region,
        endpoint_url=s.dynamodb_endpoint_url,
        config=Config(retries={"max_attempts": 5, "mode": "adaptive"}, connect_timeout=3, read_timeout=5),
    )


def table(name: str):
    return dynamodb().Table(get_settings().table(name))


def create_tables(client=None) -> list[str]:
    """Create any missing tables (local/dev/test and Postgres; AWS DynamoDB uses Terraform)."""
    s = get_settings()
    if s.database_url and client is None:
        res = dynamodb()
        res.backend.ensure_schema()
        res.backend.purge_expired()
        return []
    client = client or dynamodb().meta.client
    existing = set(client.list_tables()["TableNames"])
    created = []
    for name, spec in TABLES.items():
        full = s.table(name)
        if full in existing:
            continue
        kwargs = {
            "TableName": full,
            "KeySchema": [{"AttributeName": k, "KeyType": t} for k, t in spec["keys"]],
            "AttributeDefinitions": [{"AttributeName": a, "AttributeType": t} for a, t in spec["attrs"].items()],
            "BillingMode": "PAY_PER_REQUEST",
        }
        if spec.get("gsis"):
            kwargs["GlobalSecondaryIndexes"] = [
                {"IndexName": g["name"],
                 "KeySchema": [{"AttributeName": k, "KeyType": t} for k, t in g["keys"]],
                 "Projection": {"ProjectionType": "ALL"}}
                for g in spec["gsis"]
            ]
        try:
            client.create_table(**kwargs)
        except client.exceptions.ResourceInUseException:
            # Another uvicorn worker created it between list_tables and here.
            client.get_waiter("table_exists").wait(TableName=full)
            continue
        client.get_waiter("table_exists").wait(TableName=full)
        if spec.get("ttl"):
            try:
                client.update_time_to_live(TableName=full, TimeToLiveSpecification={
                    "Enabled": True, "AttributeName": spec["ttl"]})
            except Exception as exc:  # DynamoDB Local does not support TTL; not fatal
                logging.getLogger("db").info("TTL not enabled on %s: %s", full, type(exc).__name__)
        created.append(full)
    return created
