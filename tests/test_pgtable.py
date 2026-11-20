"""The Postgres table backend must behave like DynamoDB.

Each scenario runs the same calls against moto's DynamoDB and against
app/db/pgtable.py and compares the results (items, pagination, error codes).
Needs TEST_DATABASE_URL (a throwaway Postgres); skipped otherwise.
"""
import os
import threading
import time
from decimal import Decimal

import boto3
import pytest
from boto3.dynamodb.conditions import Attr, Key
from botocore.exceptions import ClientError

from app.core.config import get_settings
from app.db import dynamo, pgtable

PG_URL = os.getenv("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not PG_URL, reason="set TEST_DATABASE_URL to a throwaway Postgres")


@pytest.fixture
def both():
    """(moto resource, postgres resource), both empty, with the app's table definitions."""
    moto = boto3.resource("dynamodb", region_name="ap-south-1")
    dynamo.create_tables(client=moto.meta.client)
    be = pgtable.backend(PG_URL)
    be.ensure_schema()
    with be.pool.connection() as conn:
        conn.execute("TRUNCATE ak_items")
    pg = pgtable.PgResource(be, dynamo.TABLES, get_settings().table_prefix)
    return moto, pg


def tname(name):
    return get_settings().table(name)


def run(fn):
    try:
        out = fn()
        if isinstance(out, dict):
            out = {k: v for k, v in out.items() if k not in ("ResponseMetadata", "ConsumedCapacity")}
        return ("ok", out)
    except ClientError as exc:
        return ("error", exc.response["Error"]["Code"])
    except TypeError as exc:
        return ("TypeError", str(exc)[:30])


def same(moto, pg, name, fn):
    """Run fn(table) on both backends and require identical outcomes."""
    a, b = run(lambda: fn(moto.Table(tname(name)))), run(lambda: fn(pg.Table(tname(name))))
    assert a == b, f"\nmoto:     {a}\npostgres: {b}"
    return b


def test_item_roundtrip_types(both):
    moto, pg = both
    item = {"pk": "USER#1", "n": 5, "d": Decimal("2.50"), "s": "", "b": True, "z": None,
            "tags": {"rice", "maize"}, "nums": {1, 2}, "m": {"a": [1, "x", {"y": False}]}}
    same(moto, pg, "users", lambda t: t.put_item(Item=item))
    got = same(moto, pg, "users", lambda t: t.get_item(Key={"pk": "USER#1"}))
    assert got[1]["Item"]["n"] == Decimal(5) and isinstance(got[1]["Item"]["n"], Decimal)
    same(moto, pg, "users", lambda t: t.get_item(Key={"pk": "USER#missing"}))
    same(moto, pg, "users", lambda t: t.put_item(Item={"pk": "USER#2", "f": 1.5}))  # floats rejected


def test_key_validation(both):
    moto, pg = both
    same(moto, pg, "users", lambda t: t.get_item(Key={"nope": "x"}))
    same(moto, pg, "inventory", lambda t: t.get_item(Key={"owner_id": "u"}))
    same(moto, pg, "inventory", lambda t: t.put_item(Item={"owner_id": "u", "qty": 1}))


def test_conditional_writes(both):
    moto, pg = both
    put = lambda t: t.put_item(Item={"pk": "EMAIL#a", "user_id": "1"},  # noqa: E731
                               ConditionExpression="attribute_not_exists(pk)")
    same(moto, pg, "users", put)
    same(moto, pg, "users", put)  # second time: ConditionalCheckFailedException
    same(moto, pg, "users", lambda t: t.delete_item(Key={"pk": "EMAIL#zzz"}, ConditionExpression="attribute_exists(pk)"))
    same(moto, pg, "users", lambda t: t.delete_item(Key={"pk": "EMAIL#a"}, ConditionExpression="attribute_exists(pk)",
                                                     ReturnValues="ALL_OLD"))
    same(moto, pg, "users", lambda t: t.get_item(Key={"pk": "EMAIL#a"}))


def test_update_expressions_and_return_values(both):
    moto, pg = both
    k = {"pk": "LOGINFAIL#a"}
    def updated_only(resp):
        # moto also echoes key attributes for UPDATED_*; DynamoDB returns only the updated ones.
        return {"Attributes": {a: v for a, v in resp.get("Attributes", {}).items() if a != "pk"}}

    for rv in ("UPDATED_NEW", "ALL_NEW", "UPDATED_OLD", "ALL_OLD", "NONE"):
        post = updated_only if rv.startswith("UPDATED") else (lambda r: r)
        same(moto, pg, "sessions", lambda t, rv=rv, post=post: post(t.update_item(
            Key=k, UpdateExpression="ADD attempts :one SET expires_at = :exp",
            ExpressionAttributeValues={":one": 1, ":exp": 2_000_000_000}, ReturnValues=rv)))
    same(moto, pg, "sessions", lambda t: t.update_item(
        Key=k, UpdateExpression="SET #a = :z, note = :n REMOVE expires_at", ExpressionAttributeNames={"#a": "attempts"},
        ExpressionAttributeValues={":z": 0, ":n": "x"}, ReturnValues="ALL_NEW"))
    same(moto, pg, "sessions", lambda t: t.update_item(
        Key=k, UpdateExpression="SET attempts = attempts + :two, tags = if_not_exists(tags, :t)",
        ExpressionAttributeValues={":two": 2, ":t": ["a"]}, ReturnValues="ALL_NEW"))
    same(moto, pg, "sessions", lambda t: t.update_item(
        Key=k, UpdateExpression="SET pk = :x", ExpressionAttributeValues={":x": "other"}))  # key attrs immutable
    # refresh-token rotation: "used = :f" succeeds once, then fails
    same(moto, pg, "sessions", lambda t: t.put_item(Item={"pk": "REFRESH#h", "used": False, "expires_at": 9_999_999_999}))
    rotate = lambda t: t.update_item(Key={"pk": "REFRESH#h"}, UpdateExpression="SET used = :t",  # noqa: E731
                                     ConditionExpression="used = :f", ExpressionAttributeValues={":t": True, ":f": False})
    same(moto, pg, "sessions", rotate)
    same(moto, pg, "sessions", rotate)
    same(moto, pg, "users", lambda t: t.update_item(
        Key={"pk": "USER#none"}, UpdateExpression="SET a = :a", ExpressionAttributeValues={":a": 1},
        ConditionExpression="attribute_exists(pk)", ReturnValues="ALL_NEW"))
    same(moto, pg, "forum", lambda t: t.update_item(
        Key={"pk": "POST#1", "sk": "META"}, UpdateExpression="ADD likes :neg", ExpressionAttributeValues={":neg": -1},
        ReturnValues="ALL_NEW"))


def _pages(t, **kw):
    pages, start = [], None
    while True:
        args = dict(kw, **({"ExclusiveStartKey": start} if start else {}))
        resp = t.query(**args) if "KeyConditionExpression" in kw else t.scan(**args)
        pages.append(resp.get("Items", []))
        start = resp.get("LastEvaluatedKey")
        if not start:
            # A trailing empty page is harmless and backends differ on it (moto returns one only
            # when paging backwards), so compare everything up to it.
            while pages and not pages[-1]:
                pages.pop()
            return pages


def test_query_sort_order_prefix_and_pagination(both):
    moto, pg = both
    sks = ["META", "REPLY#2026-01-02#b", "REPLY#2026-01-01#a", "REPLY#2026-01-03#c", "LIKE#u1", "LIKE#U2",
           "REPLY#_x", "REPLY#éz", "REPLY#Zz"]
    for sk in sks:
        same(moto, pg, "forum", lambda t, sk=sk: t.put_item(Item={"pk": "POST#1", "sk": sk, "v": sk}))
    same(moto, pg, "forum", lambda t: t.put_item(Item={"pk": "POST#2", "sk": "REPLY#x"}))
    for fwd in (True, False):
        for limit in (None, 1, 2, 3, 100):
            kw = {"KeyConditionExpression": Key("pk").eq("POST#1") & Key("sk").begins_with("REPLY#"),
                  "ScanIndexForward": fwd}
            if limit:
                kw["Limit"] = limit
            same(moto, pg, "forum", lambda t, kw=kw: _pages(t, **kw))
    same(moto, pg, "forum", lambda t: t.query(KeyConditionExpression=Key("pk").eq("POST#1")))
    same(moto, pg, "forum", lambda t: t.query(KeyConditionExpression=Key("pk").eq("POST#1"), Select="COUNT"))
    same(moto, pg, "forum", lambda t: t.query(KeyConditionExpression=Key("pk").eq("POST#1") & Key("sk").between("L", "M")))
    same(moto, pg, "forum", lambda t: t.query(KeyConditionExpression=Key("pk").eq("POST#1") & Key("sk").gt("REPLY#2026-01-02")))
    same(moto, pg, "forum", lambda t: t.query(KeyConditionExpression=Key("pk").eq("POST#1"),
                                               FilterExpression=Attr("v").begins_with("LIKE")))
    same(moto, pg, "forum", lambda t: t.query(KeyConditionExpression=Key("pk").eq("POST#none")))


def test_gsi_query_pagination_and_sparse_items(both):
    moto, pg = both
    for i in range(7):
        same(moto, pg, "forum", lambda t, i=i: t.put_item(Item={
            "pk": f"POST#{i}", "sk": "META", "feed": "ALL", "created_at": f"2026-11-0{i % 3}T00:00:0{i}Z", "i": i}))
    # equal created_at on several items: ties are broken consistently across pages
    same(moto, pg, "forum", lambda t: t.put_item(Item={"pk": "POST#t", "sk": "META", "feed": "ALL",
                                                        "created_at": "2026-11-01T00:00:01Z"}))
    same(moto, pg, "forum", lambda t: t.put_item(Item={"pk": "POST#r", "sk": "REPLY#1", "created_at": "2026"}))  # no feed
    same(moto, pg, "forum", lambda t: t.put_item(Item={"pk": "POST#s", "sk": "META", "feed": "ALL"}))  # no range key
    for fwd in (True, False):
        for limit in (1, 2, 3, 50):
            kw = {"IndexName": "feed-index", "KeyConditionExpression": Key("feed").eq("ALL"),
                  "ScanIndexForward": fwd, "Limit": limit}
            same(moto, pg, "forum", lambda t, kw=kw: _pages(t, **kw))
    # Unknown index: DynamoDB answers ValidationException (moto says ResourceNotFoundException).
    with pytest.raises(ClientError) as exc:
        pg.Table(tname("reports")).query(IndexName="nope-index", KeyConditionExpression=Key("status").eq("x"))
    assert exc.value.response["Error"]["Code"] == "ValidationException"


def test_scan_with_filter_and_pagination(both):
    moto, pg = both
    for pk in ["USER#b", "EMAIL#a", "USER#a", "USER#c", "EMAIL#b", "USER#d"]:
        same(moto, pg, "users", lambda t, pk=pk: t.put_item(Item={"pk": pk}))

    def collect(t, limit):
        out, start = [], None
        while True:
            kw = {"FilterExpression": Attr("pk").begins_with("USER#"), "Limit": limit}
            if start:
                kw["ExclusiveStartKey"] = start
            r = t.scan(**kw)
            out += [i["pk"] for i in r["Items"]]
            start = r.get("LastEvaluatedKey")
            if not start:
                return sorted(out)  # DynamoDB scan order is unspecified; compare the set of results
    for limit in (1, 2, 4, 50):
        same(moto, pg, "users", lambda t, limit=limit: collect(t, limit))


def test_transactions(both):
    moto, pg = both
    users = tname("users")

    def register(r, uid, email):
        return r.meta.client.transact_write_items(TransactItems=[
            {"Put": {"TableName": users, "Item": {"pk": f"USER#{uid}", "id": uid},
                     "ConditionExpression": "attribute_not_exists(pk)"}},
            {"Put": {"TableName": users, "Item": {"pk": f"EMAIL#{email}", "user_id": uid},
                     "ConditionExpression": "attribute_not_exists(pk)"}},
        ])

    for args in (("1", "a@x.in"), ("2", "a@x.in"), ("3", "b@x.in")):
        a, b = run(lambda args=args: register(moto, *args)), run(lambda args=args: register(pg, *args))
        assert a[0] == b[0] and (a[0] == "ok" or a[1] == b[1]), (a, b)
    # the failed transaction wrote nothing
    same(moto, pg, "users", lambda t: t.get_item(Key={"pk": "USER#2"}))
    same(moto, pg, "users", lambda t: sorted(i["pk"] for i in t.scan()["Items"]))


def test_ttl_expired_items_are_invisible():
    be = pgtable.backend(PG_URL)
    be.ensure_schema()
    with be.pool.connection() as conn:
        conn.execute("TRUNCATE ak_items")
    t = pgtable.PgResource(be, dynamo.TABLES, get_settings().table_prefix).Table(tname("sessions"))
    past, future = int(time.time()) - 5, int(time.time()) + 600
    t.put_item(Item={"pk": "LOGINFAIL#old", "attempts": 4, "expires_at": past})
    t.put_item(Item={"pk": "LOGINFAIL#new", "attempts": 1, "expires_at": future})
    assert t.get_item(Key={"pk": "LOGINFAIL#old"}) == {}
    assert t.get_item(Key={"pk": "LOGINFAIL#new"})["Item"]["attempts"] == 1
    # an expired lockout counter restarts from zero instead of locking the farmer out forever
    got = t.update_item(Key={"pk": "LOGINFAIL#old"}, UpdateExpression="ADD attempts :one",
                        ExpressionAttributeValues={":one": 1}, ReturnValues="UPDATED_NEW")["Attributes"]
    assert got == {"attempts": 1}
    t.put_item(Item={"pk": "LOGINFAIL#gone", "expires_at": past})
    assert be.purge_expired() == 1


def test_conditional_update_is_atomic_under_concurrency():
    """Two devices racing to rotate the same refresh token: exactly one may win."""
    be = pgtable.backend(PG_URL, 10)
    be.ensure_schema()
    t = pgtable.PgResource(be, dynamo.TABLES, get_settings().table_prefix).Table(tname("sessions"))
    t.put_item(Item={"pk": "REFRESH#race", "used": False, "expires_at": int(time.time()) + 600})
    wins, losses, barrier = [], [], threading.Barrier(8)

    def attempt():
        barrier.wait()
        try:
            t.update_item(Key={"pk": "REFRESH#race"}, UpdateExpression="SET used = :t", ConditionExpression="used = :f",
                          ExpressionAttributeValues={":t": True, ":f": False})
            wins.append(1)
        except ClientError as exc:
            assert exc.response["Error"]["Code"] == "ConditionalCheckFailedException"
            losses.append(1)

    threads = [threading.Thread(target=attempt) for _ in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert (len(wins), len(losses)) == (1, 7)


def test_unsupported_parameters_fail_loudly(both):
    _, pg = both
    with pytest.raises(NotImplementedError):
        pg.Table(tname("users")).get_item(Key={"pk": "USER#1"}, ProjectionExpression="pk")
