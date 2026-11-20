"""DynamoDB-compatible tables on PostgreSQL.

Lets the API run on a plain Postgres database (for example Supabase's free
tier) without touching any route code: `table(name)` returns an object with
the subset of the boto3 `Table` API the app uses (get/put/update/delete_item,
query incl. GSIs, scan, batch_writer, load) and `dynamodb().meta.client`
supports `transact_write_items`.

Semantics kept from DynamoDB, because the routes depend on them:
  * Condition, key-condition, filter and update expressions use DynamoDB
    syntax. `Key()`/`Attr()` objects go through boto3's own expression builder.
  * Failed conditions raise botocore `ClientError` with the same error codes
    (ConditionalCheckFailedException, TransactionCanceledException).
  * Conditional writes are atomic: every write takes a per-key advisory lock
    inside a transaction, so e.g. refresh-token rotation cannot race.
  * Numbers come back as `Decimal`, Python floats are rejected (as boto3 does).
  * String keys sort by byte order (COLLATE "C"), like DynamoDB.
  * Items past their TTL attribute are treated as deleted (DynamoDB's TTL
    sweeper removes them eventually; here they are hidden immediately and
    purged periodically).

All tables share one physical table, `ak_items`, keyed by (tbl, pk, sk).
"""
from __future__ import annotations

import base64
import re
import threading
import time
from decimal import Decimal
from functools import lru_cache
from typing import Any

from boto3.dynamodb.conditions import ConditionBase, ConditionExpressionBuilder
from botocore.exceptions import ClientError

SCHEMA = """
CREATE TABLE IF NOT EXISTS ak_items (
    tbl        text COLLATE "C" NOT NULL,
    pk         text COLLATE "C" NOT NULL,
    sk         text COLLATE "C" NOT NULL DEFAULT '',
    item       jsonb NOT NULL,
    expires_at bigint,
    PRIMARY KEY (tbl, pk, sk)
);
CREATE INDEX IF NOT EXISTS ak_items_expires ON ak_items (expires_at) WHERE expires_at IS NOT NULL;
-- Small derived artefacts shared by all instances, e.g. the knowledge-base embedding index.
CREATE TABLE IF NOT EXISTS ak_blobs (
    key        text PRIMARY KEY,
    data       bytea NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now()
);
"""
PURGE_EVERY_WRITES = 500


# --- Errors ------------------------------------------------------------------

def _client_error(code: str, message: str, op: str, **extra) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": message}, **extra}, op)


def _validation(message: str, op: str) -> ClientError:
    return _client_error("ValidationException", message, op)


# --- Value encoding (Python <-> JSONB) -----------------------------------------
# DynamoDB has distinct number and set types that JSON lacks, so they are tagged.

def _enc(v: Any) -> Any:
    if v is None or isinstance(v, (bool, str)):
        return v
    if isinstance(v, float):
        raise TypeError("Float types are not supported. Use Decimal types instead.")
    if isinstance(v, (int, Decimal)):
        return {"$n": str(v)}
    if isinstance(v, (bytes, bytearray)):
        return {"$b": base64.b64encode(bytes(v)).decode()}
    if isinstance(v, (set, frozenset)):
        if not v:
            raise _validation("An string set may not be empty", "PutItem")
        first = next(iter(v))
        if isinstance(first, str):
            return {"$ss": sorted(v)}
        if isinstance(first, (bytes, bytearray)):
            return {"$bs": sorted(base64.b64encode(bytes(x)).decode() for x in v)}
        return {"$ns": sorted(str(x) for x in v)}
    if isinstance(v, (list, tuple)):
        return [_enc(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _enc(x) for k, x in v.items()}
    raise TypeError(f"Unsupported type {type(v).__name__} for DynamoDB value")


def _dec(v: Any) -> Any:
    if isinstance(v, list):
        return [_dec(x) for x in v]
    if isinstance(v, dict):
        if len(v) == 1:
            (tag, val), = v.items()
            if tag == "$n":
                return Decimal(val)
            if tag == "$ss":
                return set(val)
            if tag == "$ns":
                return {Decimal(x) for x in val}
            if tag == "$b":
                return base64.b64decode(val)
            if tag == "$bs":
                return {base64.b64decode(x) for x in val}
        return {k: _dec(x) for k, x in v.items()}
    return v


def _norm(v: Any) -> Any:
    """Normalise numbers so 1 == Decimal('1') == Decimal('1.0') in comparisons."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, Decimal)):
        return Decimal(v)
    return v


# --- Expression parsing ------------------------------------------------------

_TOKEN = re.compile(r"\s*(?:(?P<op><>|<=|>=|=|<|>|\(|\)|,|\[|\]|\.|\+|-)|(?P<name>#[A-Za-z0-9_]+)"
                    r"|(?P<value>:[A-Za-z0-9_]+)|(?P<num>\d+)|(?P<word>[A-Za-z_][A-Za-z0-9_]*))")
_KEYWORDS = {"AND", "OR", "NOT", "BETWEEN", "IN", "SET", "ADD", "REMOVE", "DELETE"}


def _tokenize(expr: str) -> list[tuple[str, str]]:
    out, pos = [], 0
    expr = expr.rstrip()
    while pos < len(expr):
        m = _TOKEN.match(expr, pos)
        if not m or m.end() == pos:
            raise _validation(f"Invalid expression near: {expr[pos:pos + 20]!r}", "Expression")
        kind = m.lastgroup
        text = m.group(kind)
        if kind == "word" and text.upper() in _KEYWORDS:
            kind, text = "kw", text.upper()
        out.append((kind, text))
        pos = m.end()
    return out


class _Parser:
    def __init__(self, expr: str, names: dict | None, values: dict | None):
        self.toks, self.i = _tokenize(expr), 0
        self.names, self.values = names or {}, values or {}

    # token helpers
    def peek(self, k: int = 0):
        j = self.i + k
        return self.toks[j] if j < len(self.toks) else (None, None)

    def take(self, kind=None, text=None):
        tok = self.peek()
        if (kind and tok[0] != kind) or (text and tok[1] != text):
            raise _validation(f"Invalid expression: expected {text or kind}, got {tok[1]!r}", "Expression")
        self.i += 1
        return tok

    def at(self, kind=None, text=None) -> bool:
        tok = self.peek()
        return tok[0] is not None and (kind is None or tok[0] == kind) and (text is None or tok[1] == text)

    def done(self) -> bool:
        return self.i >= len(self.toks)

    # paths and operands
    def path(self) -> tuple:
        parts = [self._name()]
        while self.at("op", ".") or self.at("op", "["):
            if self.take()[1] == ".":
                parts.append(self._name())
            else:
                parts.append(int(self.take("num")[1]))
                self.take("op", "]")
        return ("path", tuple(parts))

    def _name(self) -> str:
        kind, text = self.take()
        if kind == "name":
            if text not in self.names:
                raise _validation(f"Missing ExpressionAttributeNames entry for {text}", "Expression")
            return self.names[text]
        if kind == "word":
            return text
        raise _validation(f"Invalid attribute name {text!r}", "Expression")

    def operand(self) -> tuple:
        if self.at("value"):
            ref = self.take()[1]
            if ref not in self.values:
                raise _validation(f"Missing ExpressionAttributeValues entry for {ref}", "Expression")
            return ("value", self.values[ref])
        if self.at("word") and self.peek(1) == ("op", "("):
            fn = self.take()[1]
            self.take("op", "(")
            args = [self.operand()]
            while self.at("op", ","):
                self.take()
                args.append(self.operand())
            self.take("op", ")")
            return ("fn", fn, args)
        return self.path()

    # conditions
    def condition(self) -> tuple:
        node = self._and()
        while self.at("kw", "OR"):
            self.take()
            node = ("or", node, self._and())
        return node

    def _and(self) -> tuple:
        node = self._not()
        while self.at("kw", "AND"):
            self.take()
            node = ("and", node, self._not())
        return node

    def _not(self) -> tuple:
        if self.at("kw", "NOT"):
            self.take()
            return ("not", self._not())
        return self._primary()

    def _primary(self) -> tuple:
        if self.at("op", "("):
            self.take()
            node = self.condition()
            self.take("op", ")")
            return node
        left = self.operand()
        if self.at("op") and self.peek()[1] in ("=", "<>", "<", "<=", ">", ">="):
            op = self.take()[1]
            return ("cmp", op, left, self.operand())
        if self.at("kw", "BETWEEN"):
            self.take()
            lo = self.operand()
            self.take("kw", "AND")
            return ("between", left, lo, self.operand())
        if self.at("kw", "IN"):
            self.take()
            self.take("op", "(")
            opts = [self.operand()]
            while self.at("op", ","):
                self.take()
                opts.append(self.operand())
            self.take("op", ")")
            return ("in", left, opts)
        if left[0] == "fn":
            return ("call", left)
        raise _validation("Invalid condition expression", "Expression")

    # update expressions
    def update(self) -> list[tuple]:
        actions = []
        while not self.done():
            clause = self.take("kw")[1]
            while True:
                if clause == "SET":
                    target = self.path()
                    self.take("op", "=")
                    val = self.operand()
                    if self.at("op", "+") or self.at("op", "-"):
                        val = ("arith", self.take()[1], val, self.operand())
                    actions.append(("SET", target, val))
                elif clause == "REMOVE":
                    actions.append(("REMOVE", self.path()))
                elif clause in ("ADD", "DELETE"):
                    target = self.path()
                    actions.append((clause, target, self.operand()))
                else:
                    raise _validation(f"Unsupported update clause {clause}", "UpdateItem")
                if self.at("op", ","):
                    self.take()
                    continue
                break
        return actions


_MISSING = object()


def _get_path(item: dict | None, path: tuple) -> Any:
    cur: Any = item
    for p in path:
        if isinstance(p, int):
            if not isinstance(cur, list) or p >= len(cur):
                return _MISSING
            cur = cur[p]
        else:
            if not isinstance(cur, dict) or p not in cur:
                return _MISSING
            cur = cur[p]
    return cur


def _set_path(item: dict, path: tuple, value: Any) -> None:
    cur: Any = item
    for p in path[:-1]:
        cur = cur[p]
    cur[path[-1]] = value


def _del_path(item: dict, path: tuple) -> None:
    cur: Any = item
    for p in path[:-1]:
        cur = cur.get(p) if isinstance(cur, dict) else (cur[p] if isinstance(cur, list) and p < len(cur) else None)
        if cur is None:
            return
    if isinstance(cur, dict):
        cur.pop(path[-1], None)
    elif isinstance(cur, list) and path[-1] < len(cur):
        cur.pop(path[-1])


def _eval_operand(node: tuple, item: dict | None) -> Any:
    kind = node[0]
    if kind == "value":
        return node[1]
    if kind == "path":
        return _get_path(item, node[1])
    if kind == "fn":
        fn, args = node[1], node[2]
        if fn == "size":
            v = _eval_operand(args[0], item)
            return _MISSING if v is _MISSING else Decimal(len(v))
        if fn == "if_not_exists":
            v = _eval_operand(args[0], item)
            return _eval_operand(args[1], item) if v is _MISSING else v
        if fn == "list_append":
            a, b = _eval_operand(args[0], item), _eval_operand(args[1], item)
            return list(a) + list(b)
        raise _validation(f"Unsupported function {fn}", "Expression")
    if kind == "arith":
        a, b = _eval_operand(node[2], item), _eval_operand(node[3], item)
        if a is _MISSING or b is _MISSING:
            raise _validation("The provided expression refers to an attribute that does not exist in the item",
                              "UpdateItem")
        return Decimal(a) + Decimal(b) if node[1] == "+" else Decimal(a) - Decimal(b)
    raise _validation("Invalid operand", "Expression")


def _compare(op: str, a: Any, b: Any) -> bool:
    if a is _MISSING or b is _MISSING:
        return op == "<>" and not (a is _MISSING and b is _MISSING)
    a, b = _norm(a), _norm(b)
    if op == "=":
        return a == b
    if op == "<>":
        return a != b
    if type(a) is not type(b) or isinstance(a, (bool, dict, list, set)):
        return False
    return {"<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b}[op]


def _eval_cond(node: tuple, item: dict | None) -> bool:
    kind = node[0]
    if kind == "and":
        return _eval_cond(node[1], item) and _eval_cond(node[2], item)
    if kind == "or":
        return _eval_cond(node[1], item) or _eval_cond(node[2], item)
    if kind == "not":
        return not _eval_cond(node[1], item)
    if kind == "cmp":
        return _compare(node[1], _eval_operand(node[2], item), _eval_operand(node[3], item))
    if kind == "between":
        v = _eval_operand(node[1], item)
        return _compare(">=", v, _eval_operand(node[2], item)) and _compare("<=", v, _eval_operand(node[3], item))
    if kind == "in":
        v = _eval_operand(node[1], item)
        return any(_compare("=", v, _eval_operand(o, item)) for o in node[2])
    if kind == "call":
        fn, args = node[1][1], node[1][2]
        if fn == "attribute_exists":
            return _eval_operand(args[0], item) is not _MISSING
        if fn == "attribute_not_exists":
            return _eval_operand(args[0], item) is _MISSING
        if fn == "begins_with":
            v, prefix = _eval_operand(args[0], item), _eval_operand(args[1], item)
            return isinstance(v, (str, bytes)) and type(v) is type(prefix) and v.startswith(prefix)
        if fn == "contains":
            v, x = _eval_operand(args[0], item), _eval_operand(args[1], item)
            if isinstance(v, str):
                return isinstance(x, str) and x in v
            if isinstance(v, (set, list)):
                return _norm(x) in {_norm(e) for e in v} if isinstance(v, set) else any(
                    _compare("=", e, x) for e in v)
            return False
        if fn == "attribute_type":
            raise _validation("attribute_type is not supported", "Expression")
        raise _validation(f"Unsupported function {fn}", "Expression")
    raise _validation("Invalid condition", "Expression")


def _build(cond, names: dict | None, values: dict | None, builder: ConditionExpressionBuilder,
           is_key: bool = False) -> tuple[str, dict, dict]:
    """Turn a Key()/Attr() object or an expression string into (expr, names, values)."""
    names, values = dict(names or {}), dict(values or {})
    if isinstance(cond, ConditionBase):
        built = builder.build_expression(cond, is_key_condition=is_key)
        names.update(built.attribute_name_placeholders)
        values.update(built.attribute_value_placeholders)
        return built.condition_expression, names, values
    return cond, names, values


def _parse_condition(cond, names, values, builder=None, is_key=False) -> tuple:
    expr, n, v = _build(cond, names, values, builder or ConditionExpressionBuilder(), is_key)
    p = _Parser(expr, n, v)
    node = p.condition()
    if not p.done():
        raise _validation(f"Invalid expression: unexpected {p.peek()[1]!r}", "Expression")
    return node


def _apply_update(item: dict, actions: list[tuple], key_attrs: set[str]) -> set[str]:
    """Apply parsed update actions in place; returns the top-level attributes touched."""
    touched = set()
    for action in actions:
        target = action[1][1]
        if target[0] in key_attrs:
            raise _validation("Cannot update attribute " + target[0] + ". This attribute is part of the key",
                              "UpdateItem")
        touched.add(target[0])
        if action[0] == "SET":
            _set_path(item, target, _eval_operand(action[2], item))
        elif action[0] == "REMOVE":
            _del_path(item, target)
        elif action[0] == "ADD":
            cur, add = _get_path(item, target), _eval_operand(action[2], item)
            if isinstance(add, (set, frozenset)):
                _set_path(item, target, (set() if cur is _MISSING else set(cur)) | set(add))
            elif isinstance(add, (int, Decimal)) and not isinstance(add, bool):
                if cur is not _MISSING and (isinstance(cur, bool) or not isinstance(cur, (int, Decimal))):
                    raise _validation("An operand in the update expression has an incorrect data type",
                                      "UpdateItem")
                _set_path(item, target, (Decimal(0) if cur is _MISSING else Decimal(cur)) + Decimal(add))
            else:
                raise _validation("ADD supports only numbers and sets", "UpdateItem")
        elif action[0] == "DELETE":
            cur, rem = _get_path(item, target), _eval_operand(action[2], item)
            if cur is not _MISSING:
                left = set(cur) - set(rem)
                if left:
                    _set_path(item, target, left)
                else:
                    _del_path(item, target)
    return touched


# --- Database ------------------------------------------------------------------

class _Backend:
    """Connection pool + schema. One per database URL, shared by every table object."""

    def __init__(self, url: str, pool_size: int):
        from psycopg_pool import ConnectionPool
        self.pool = ConnectionPool(
            url, min_size=1, max_size=pool_size, open=True, name="ameer-kisaan",
            # prepare_threshold=None: safe behind PgBouncer/Supavisor in transaction mode.
            kwargs={"autocommit": True, "prepare_threshold": None},
        )
        self._writes = 0
        self._lock = threading.Lock()
        self._schema_ready = False

    def ensure_schema(self) -> None:
        with self.pool.connection() as conn, conn.transaction():
            conn.execute("SELECT pg_advisory_xact_lock(4242)")  # one creator at a time across workers
            conn.execute(SCHEMA)
        self._schema_ready = True

    def purge_expired(self) -> int:
        with self.pool.connection() as conn:
            return conn.execute("DELETE FROM ak_items WHERE expires_at < %s", (int(time.time()),)).rowcount

    def blob_get(self, key: str) -> bytes | None:
        self.ensure_schema_once()
        with self.pool.connection() as conn:
            row = conn.execute("SELECT data FROM ak_blobs WHERE key = %s", (key,)).fetchone()
        return bytes(row[0]) if row else None

    def blob_put(self, key: str, data: bytes) -> None:
        self.ensure_schema_once()
        with self.pool.connection() as conn:
            conn.execute("INSERT INTO ak_blobs (key, data) VALUES (%s, %s) ON CONFLICT (key) DO UPDATE "
                         "SET data = EXCLUDED.data, updated_at = now()", (key, data))

    def ensure_schema_once(self) -> None:
        if not self._schema_ready:
            self.ensure_schema()

    def note_write(self) -> None:
        with self._lock:
            self._writes += 1
            due = self._writes % PURGE_EVERY_WRITES == 0
        if due:
            self.purge_expired()


@lru_cache
def backend(url: str, pool_size: int = 5) -> _Backend:
    return _Backend(url, pool_size)


def _jsonb(item: dict):
    from psycopg.types.json import Jsonb
    return Jsonb(_enc(item))


class _Spec:
    def __init__(self, full_name: str, spec: dict):
        self.name = full_name
        keys = spec["keys"]
        self.hash_key = keys[0][0]
        self.range_key = keys[1][0] if len(keys) > 1 else None
        self.ttl = spec.get("ttl")
        self.gsis = {g["name"]: (g["keys"][0][0], g["keys"][1][0] if len(g["keys"]) > 1 else None)
                     for g in spec.get("gsis", [])}
        for attr, typ in spec["attrs"].items():
            if typ != "S":
                raise NotImplementedError(f"{full_name}.{attr}: only string key attributes are supported")

    @property
    def key_attrs(self) -> set[str]:
        return {self.hash_key} | ({self.range_key} if self.range_key else set())

    def key_of(self, key: dict, op: str) -> tuple[str, str]:
        if set(key) != self.key_attrs:
            raise _validation("The provided key element does not match the schema", op)
        pk, sk = key[self.hash_key], key[self.range_key] if self.range_key else ""
        if not isinstance(pk, str) or not isinstance(sk, str) or pk == "" or (self.range_key and sk == ""):
            raise _validation("One or more parameter values are not valid. Key values must be non-empty strings",
                              op)
        return pk, sk

    def expiry(self, item: dict) -> int | None:
        v = item.get(self.ttl) if self.ttl else None
        return int(v) if isinstance(v, (int, Decimal)) and not isinstance(v, bool) else None


def _now() -> int:
    return int(time.time())


def _alive_sql() -> str:
    return "(expires_at IS NULL OR expires_at >= %s)"


class _WriteTxn:
    """A transaction holding per-key locks; reads see the locked current item."""

    def __init__(self, conn):
        self.conn = conn

    def lock(self, tbl: str, pk: str, sk: str) -> None:
        self.conn.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (f"{tbl}\x1f{pk}\x1f{sk}",))

    def current(self, tbl: str, pk: str, sk: str) -> dict | None:
        row = self.conn.execute(
            "SELECT item FROM ak_items WHERE tbl = %s AND pk = %s AND sk = %s "
            "AND (expires_at IS NULL OR expires_at >= %s)",
            (tbl, pk, sk, _now())).fetchone()
        return _dec(row[0]) if row else None

    def upsert(self, spec: _Spec, pk: str, sk: str, item: dict) -> None:
        self.conn.execute(
            "INSERT INTO ak_items (tbl, pk, sk, item, expires_at) VALUES (%s, %s, %s, %s, %s) "
            "ON CONFLICT (tbl, pk, sk) DO UPDATE SET item = EXCLUDED.item, expires_at = EXCLUDED.expires_at",
            (spec.name, pk, sk, _jsonb(item), spec.expiry(item)))

    def delete(self, tbl: str, pk: str, sk: str) -> None:
        self.conn.execute("DELETE FROM ak_items WHERE tbl = %s AND pk = %s AND sk = %s", (tbl, pk, sk))


_PUT_KW = {"Item", "ConditionExpression", "ExpressionAttributeNames", "ExpressionAttributeValues", "ReturnValues"}
_GET_KW = {"Key", "ConsistentRead"}
_UPDATE_KW = {"Key", "UpdateExpression", "ConditionExpression", "ExpressionAttributeNames",
              "ExpressionAttributeValues", "ReturnValues"}
_DELETE_KW = {"Key", "ConditionExpression", "ExpressionAttributeNames", "ExpressionAttributeValues", "ReturnValues"}
_QUERY_KW = {"KeyConditionExpression", "IndexName", "ScanIndexForward", "Limit", "ExclusiveStartKey", "Select",
             "FilterExpression", "ExpressionAttributeNames", "ExpressionAttributeValues", "ConsistentRead"}
_SCAN_KW = {"FilterExpression", "Limit", "ExclusiveStartKey", "Select", "ExpressionAttributeNames",
            "ExpressionAttributeValues", "ConsistentRead"}


def _check_kwargs(kwargs: dict, allowed: set[str], op: str) -> None:
    unknown = set(kwargs) - allowed
    if unknown:
        raise NotImplementedError(f"{op}: unsupported parameters {sorted(unknown)}")


class PgTable:
    def __init__(self, be: _Backend, spec: _Spec):
        self._be, self._spec = be, spec
        self.name = self.table_name = spec.name

    # -- single-item operations --
    def load(self) -> None:
        with self._be.pool.connection() as conn:
            conn.execute("SELECT 1 FROM ak_items LIMIT 1")

    def get_item(self, **kw) -> dict:
        _check_kwargs(kw, _GET_KW, "GetItem")
        pk, sk = self._spec.key_of(kw["Key"], "GetItem")
        with self._be.pool.connection() as conn:
            row = conn.execute(
                "SELECT item FROM ak_items WHERE tbl = %s AND pk = %s AND sk = %s "
            "AND (expires_at IS NULL OR expires_at >= %s)",
                (self.name, pk, sk, _now())).fetchone()
        return {"Item": _dec(row[0])} if row else {}

    def put_item(self, **kw) -> dict:
        _check_kwargs(kw, _PUT_KW, "PutItem")
        item = kw["Item"]
        _enc(item)  # type check before touching the database
        pk, sk = self._spec.key_of({k: item.get(k) for k in self._spec.key_attrs}, "PutItem")
        cond = (_parse_condition(kw["ConditionExpression"], kw.get("ExpressionAttributeNames"),
                                 kw.get("ExpressionAttributeValues"))
                if kw.get("ConditionExpression") else None)
        with self._be.pool.connection() as conn, conn.transaction():
            tx = _WriteTxn(conn)
            tx.lock(self.name, pk, sk)
            old = tx.current(self.name, pk, sk)
            if cond is not None and not _eval_cond(cond, old):
                raise _client_error("ConditionalCheckFailedException", "The conditional request failed", "PutItem")
            tx.upsert(self._spec, pk, sk, item)
        self._be.note_write()
        return {"Attributes": old} if kw.get("ReturnValues") == "ALL_OLD" and old else {}

    def update_item(self, **kw) -> dict:
        _check_kwargs(kw, _UPDATE_KW, "UpdateItem")
        key = kw["Key"]
        pk, sk = self._spec.key_of(key, "UpdateItem")
        names, values = kw.get("ExpressionAttributeNames"), kw.get("ExpressionAttributeValues")
        p = _Parser(kw["UpdateExpression"], names, values)
        actions = p.update()
        cond = _parse_condition(kw["ConditionExpression"], names, values) if kw.get("ConditionExpression") else None
        rv = kw.get("ReturnValues", "NONE")
        with self._be.pool.connection() as conn, conn.transaction():
            tx = _WriteTxn(conn)
            tx.lock(self.name, pk, sk)
            old = tx.current(self.name, pk, sk)
            if cond is not None and not _eval_cond(cond, old):
                raise _client_error("ConditionalCheckFailedException", "The conditional request failed",
                                    "UpdateItem")
            new = _dec(_enc(old)) if old else dict(key)
            touched = _apply_update(new, actions, self._spec.key_attrs)
            new = _dec(_enc(new))  # normalise: numbers as Decimal, as DynamoDB returns them
            tx.upsert(self._spec, pk, sk, new)
        self._be.note_write()
        if rv == "ALL_NEW":
            return {"Attributes": new}
        if rv in ("UPDATED_NEW", "UPDATED_OLD"):
            # Only attributes whose value actually changed (or were added/removed).
            touched = {k for k in touched if (old or {}).get(k, _MISSING) != new.get(k, _MISSING)}
        if rv == "UPDATED_NEW":
            return {"Attributes": {k: new[k] for k in touched if k in new}}
        if rv == "ALL_OLD":
            return {"Attributes": old} if old else {}
        if rv == "UPDATED_OLD":
            return {"Attributes": {k: old[k] for k in touched if old and k in old}} if old else {}
        return {}

    def delete_item(self, **kw) -> dict:
        _check_kwargs(kw, _DELETE_KW, "DeleteItem")
        pk, sk = self._spec.key_of(kw["Key"], "DeleteItem")
        cond = (_parse_condition(kw["ConditionExpression"], kw.get("ExpressionAttributeNames"),
                                 kw.get("ExpressionAttributeValues"))
                if kw.get("ConditionExpression") else None)
        with self._be.pool.connection() as conn, conn.transaction():
            tx = _WriteTxn(conn)
            tx.lock(self.name, pk, sk)
            old = tx.current(self.name, pk, sk)
            if cond is not None and not _eval_cond(cond, old):
                raise _client_error("ConditionalCheckFailedException", "The conditional request failed",
                                    "DeleteItem")
            tx.delete(self.name, pk, sk)
        return {"Attributes": old} if kw.get("ReturnValues") == "ALL_OLD" and old else {}

    # -- multi-item reads --
    def query(self, **kw) -> dict:
        _check_kwargs(kw, _QUERY_KW, "Query")
        builder = ConditionExpressionBuilder()
        names, values = kw.get("ExpressionAttributeNames"), kw.get("ExpressionAttributeValues")
        key_node = _parse_condition(kw["KeyConditionExpression"], names, values, builder, is_key=True)
        flt = (_parse_condition(kw["FilterExpression"], names, values, builder)
               if kw.get("FilterExpression") else None)

        index = kw.get("IndexName")
        if index:
            if index not in self._spec.gsis:
                raise _validation("The table does not have the specified index: " + index, "Query")
            hk, rk = self._spec.gsis[index]
        else:
            hk, rk = self._spec.hash_key, self._spec.range_key

        def col(attr: str) -> str:
            if not index and attr == self._spec.hash_key:
                return "pk"
            if not index and attr == self._spec.range_key:
                return "sk"
            return f"(item->>'{attr}') COLLATE \"C\""  # attr comes from the table spec, never user input

        hash_val, range_cond = self._split_key_condition(key_node, hk, rk)
        where = ["tbl = %s", f"{col(hk)} = %s", _alive_sql()]
        args: list = [self.name, hash_val, _now()]
        if index and rk:
            where.append(f"{col(rk)} IS NOT NULL")  # GSIs are sparse: items without the range key are not indexed
        if range_cond:
            op, operands = range_cond
            c = col(rk)
            if op == "begins_with":
                where.append(f"starts_with({c}, %s)")
                args.append(operands[0])
            elif op == "between":
                where.append(f"{c} BETWEEN %s AND %s")
                args += operands
            else:
                where.append(f"{c} {op} %s")
                args.append(operands[0])

        forward = kw.get("ScanIndexForward", True)
        direction = "ASC" if forward else "DESC"
        order_cols = [col(rk)] if rk else []
        if index:
            order_cols += ["pk", "sk"]
        start = kw.get("ExclusiveStartKey")
        if start and order_cols:
            start_vals = ([start[rk]] if rk else []) + ([start[self._spec.hash_key],
                                                         start.get(self._spec.range_key, "") if self._spec.range_key
                                                         else ""] if index else [])
            where.append(f"({', '.join(order_cols)}) {'>' if forward else '<'} ({', '.join(['%s'] * len(start_vals))})")
            args += start_vals
        order = f"ORDER BY {', '.join(f'{c} {direction}' for c in order_cols)}" if order_cols else ""
        return self._fetch(where, args, order, kw.get("Limit"), flt, kw.get("Select"),
                           index_keys=(hk, rk) if index else None)

    def scan(self, **kw) -> dict:
        _check_kwargs(kw, _SCAN_KW, "Scan")
        flt = (_parse_condition(kw["FilterExpression"], kw.get("ExpressionAttributeNames"),
                                kw.get("ExpressionAttributeValues"))
               if kw.get("FilterExpression") else None)
        where, args = ["tbl = %s", _alive_sql()], [self.name, _now()]
        start = kw.get("ExclusiveStartKey")
        if start:
            where.append("(pk, sk) > (%s, %s)")
            args += [start[self._spec.hash_key], start.get(self._spec.range_key, "") if self._spec.range_key else ""]
        return self._fetch(where, args, "ORDER BY pk ASC, sk ASC", kw.get("Limit"), flt, kw.get("Select"),
                           index_keys=None)

    def _split_key_condition(self, node: tuple, hk: str, rk: str | None):
        parts = [node] if node[0] != "and" else [node[1], node[2]]
        if node[0] == "and" and (node[1][0] == "and" or node[2][0] == "and"):
            raise _validation("Query key condition not supported", "Query")
        hash_val, range_cond = None, None
        for part in parts:
            if part[0] == "cmp" and part[2][0] == "path" and part[2][1] == (hk,) and part[1] == "=" \
                    and part[3][0] == "value":
                hash_val = part[3][1]
            elif rk and part[0] == "cmp" and part[2][0] == "path" and part[2][1] == (rk,) and part[3][0] == "value" \
                    and part[1] in ("=", "<", "<=", ">", ">="):
                range_cond = (part[1], [part[3][1]])
            elif rk and part[0] == "between" and part[1][0] == "path" and part[1][1] == (rk,):
                range_cond = ("between", [part[2][1], part[3][1]])
            elif rk and part[0] == "call" and part[1][1] == "begins_with" and part[1][2][0][1] == (rk,):
                range_cond = ("begins_with", [part[1][2][1][1]])
            else:
                raise _validation("Query key condition not supported", "Query")
        if not isinstance(hash_val, str):
            raise _validation("Query condition missed key schema element: " + hk, "Query")
        return hash_val, range_cond

    def _fetch(self, where, args, order, limit, flt, select, index_keys):
        # Only fixed SQL fragments and column names from the table spec are interpolated here;
        # every caller-supplied value (keys, cursors, prefixes) is a bound %s parameter.
        sql = f"SELECT pk, sk, item FROM ak_items WHERE {' AND '.join(where)} {order}"  # noqa: S608
        if limit is not None:
            if not isinstance(limit, int) or limit < 1:
                raise _validation("Limit must be a positive integer", "Query")
            sql += " LIMIT %s"
            args = [*args, limit + 1]  # one extra row tells us whether another page exists
        with self._be.pool.connection() as conn:
            rows = conn.execute(sql, args).fetchall()
        more = limit is not None and len(rows) > limit
        rows = rows[:limit] if limit is not None else rows
        items = [_dec(r[2]) for r in rows]
        scanned = len(items)
        if flt is not None:
            items = [i for i in items if _eval_cond(flt, i)]
        out: dict = {"Count": len(items), "ScannedCount": scanned}
        if select != "COUNT":
            out["Items"] = items
        # Unlike DynamoDB, no cursor is returned when the page happens to end exactly at the
        # last item, so clients never fetch an empty trailing page.
        if more:
            last = _dec(rows[-1][2])
            lek = {self._spec.hash_key: rows[-1][0]}
            if self._spec.range_key:
                lek[self._spec.range_key] = rows[-1][1]
            if index_keys:
                for attr in index_keys:
                    if attr:
                        lek[attr] = last[attr]
            out["LastEvaluatedKey"] = lek
        return out

    def batch_writer(self, overwrite_by_pkeys=None):
        return _BatchWriter(self)


class _BatchWriter:
    def __init__(self, t: PgTable):
        self.t = t

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def put_item(self, Item):  # noqa: N803 - boto3 signature
        self.t.put_item(Item=Item)

    def delete_item(self, Key):  # noqa: N803 - boto3 signature
        self.t.delete_item(Key=Key)


class PgClient:
    """The few low-level client calls the app makes on `dynamodb().meta.client`."""

    def __init__(self, resource: PgResource):
        self._r = resource

    def transact_write_items(self, TransactItems):  # noqa: N803 - boto3 signature
        if not 1 <= len(TransactItems) <= 100:
            raise _validation("Transactions must contain between 1 and 100 items", "TransactWriteItems")
        plans = []
        for entry in TransactItems:
            (op, body), = entry.items()
            t = self._r.Table(body["TableName"])
            key = ({k: body["Item"].get(k) for k in t._spec.key_attrs} if op == "Put" else body["Key"])
            pk, sk = t._spec.key_of(key, "TransactWriteItems")
            cond = (_parse_condition(body["ConditionExpression"], body.get("ExpressionAttributeNames"),
                                     body.get("ExpressionAttributeValues"))
                    if body.get("ConditionExpression") else None)
            plans.append((op, body, t, pk, sk, cond))
        keys = [(t.name, pk, sk) for _, _, t, pk, sk, _ in plans]
        if len(set(keys)) != len(keys):
            raise _validation("Transaction request cannot include multiple operations on one item",
                              "TransactWriteItems")
        be = self._r.backend
        with be.pool.connection() as conn, conn.transaction():
            tx = _WriteTxn(conn)
            for k in sorted(keys):  # fixed lock order: no deadlocks between concurrent transactions
                tx.lock(*k)
            reasons, olds = [], []
            for _op, _body, t, pk, sk, cond in plans:
                old = tx.current(t.name, pk, sk)
                olds.append(old)
                ok = cond is None or _eval_cond(cond, old)
                reasons.append({"Code": "None"} if ok else {"Code": "ConditionalCheckFailed",
                                                             "Message": "The conditional request failed"})
            if any(r["Code"] != "None" for r in reasons):
                raise _client_error("TransactionCanceledException",
                                    "Transaction cancelled, please refer cancellation reasons for specific reasons",
                                    "TransactWriteItems", CancellationReasons=reasons)
            for (op, body, t, pk, sk, _), old in zip(plans, olds, strict=True):
                if op == "Put":
                    _enc(body["Item"])
                    tx.upsert(t._spec, pk, sk, body["Item"])
                elif op == "Delete":
                    tx.delete(t.name, pk, sk)
                elif op == "Update":
                    p = _Parser(body["UpdateExpression"], body.get("ExpressionAttributeNames"),
                                body.get("ExpressionAttributeValues"))
                    new = _dec(_enc(old)) if old else dict(body["Key"])
                    _apply_update(new, p.update(), t._spec.key_attrs)
                    tx.upsert(t._spec, pk, sk, new)
                elif op != "ConditionCheck":
                    raise _validation(f"Unsupported transaction operation {op}", "TransactWriteItems")
        be.note_write()
        return {}


class _Meta:
    def __init__(self, client):
        self.client = client


class PgResource:
    """Stands in for `boto3.resource("dynamodb")`."""

    def __init__(self, be: _Backend, specs: dict[str, dict], prefix: str):
        self.backend, self._specs, self._prefix = be, specs, prefix
        self.meta = _Meta(PgClient(self))

    def Table(self, name: str) -> PgTable:  # noqa: N802 - boto3 signature
        logical = name[len(self._prefix):] if name.startswith(self._prefix) else name
        if logical not in self._specs:
            raise _client_error("ResourceNotFoundException", f"Requested resource not found: {name}", "DescribeTable")
        return PgTable(self.backend, _Spec(name, self._specs[logical]))
