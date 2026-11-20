"""Community forum: posts, replies and likes shared between all users."""
import base64
import json
import uuid
from datetime import datetime, timezone

from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Response

from app.core import audit
from app.core.deps import CurrentUser, get_current_user, rate_limit
from app.db.dynamo import table
from app.schemas import PostIn, PostOut, PostPage, ReplyOut
from app.services.users import display_name

router = APIRouter(prefix="/forum", tags=["forum"])


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _encode_cursor(key: dict | None) -> str | None:
    return base64.urlsafe_b64encode(json.dumps(key).encode()).decode() if key else None


def _decode_cursor(cursor: str | None) -> dict | None:
    if not cursor:
        return None
    try:
        key = json.loads(base64.urlsafe_b64decode(cursor.encode()))
        if set(key) != {"pk", "sk", "feed", "created_at"} or not all(isinstance(v, str) for v in key.values()):
            raise ValueError
        return key
    except Exception:
        raise HTTPException(400, "Invalid cursor")


def _get_post(post_id: str) -> dict:
    item = table("forum").get_item(Key={"pk": f"POST#{post_id}", "sk": "META"}).get("Item")
    if not item or item.get("deleted"):
        raise HTTPException(404, "Post not found")
    return item


def _replies(post_id: str) -> list[ReplyOut]:
    resp = table("forum").query(KeyConditionExpression=Key("pk").eq(f"POST#{post_id}") & Key("sk").begins_with("REPLY#"))
    return [ReplyOut(id=r["reply_id"], text=r["text"], author_id=r["author_id"], author_name=r["author_name"],
                     created_at=r["created_at"]) for r in resp.get("Items", [])]


def _liked(post_id: str, user_id: str) -> bool:
    return "Item" in table("forum").get_item(Key={"pk": f"POST#{post_id}", "sk": f"LIKE#{user_id}"})


def _out(p: dict, user_id: str, with_replies: bool = True) -> PostOut:
    return PostOut(
        id=p["post_id"], text=p["text"], author_id=p["author_id"], author_name=p["author_name"],
        likes=int(p.get("likes", 0)), reply_count=int(p.get("reply_count", 0)), created_at=p["created_at"],
        edited_at=p.get("edited_at"), liked_by_me=_liked(p["post_id"], user_id),
        replies=_replies(p["post_id"]) if with_replies else [],
    )


@router.get("/posts", response_model=PostPage)
def list_posts(limit: int = 20, cursor: str | None = None, user: CurrentUser = Depends(get_current_user)):
    kwargs = {"IndexName": "feed-index", "KeyConditionExpression": Key("feed").eq("ALL"),
              "ScanIndexForward": False, "Limit": min(max(limit, 1), 50)}
    start = _decode_cursor(cursor)
    if start:
        kwargs["ExclusiveStartKey"] = start
    resp = table("forum").query(**kwargs)
    return PostPage(items=[_out(p, user.id) for p in resp.get("Items", [])],
                    next_cursor=_encode_cursor(resp.get("LastEvaluatedKey")))


@router.post("/posts", response_model=PostOut, status_code=201,
             dependencies=[Depends(rate_limit("forum-post", 20, 300))])
def create_post(body: PostIn, bg: BackgroundTasks, user: CurrentUser = Depends(get_current_user)):
    post_id, now = uuid.uuid4().hex, _now()
    item = {"pk": f"POST#{post_id}", "sk": "META", "feed": "ALL", "created_at": now, "post_id": post_id,
            "text": body.text, "author_id": user.id, "author_name": display_name(user.id), "likes": 0,
            "reply_count": 0}
    table("forum").put_item(Item=item)
    bg.add_task(audit.record, "forum.post", user.id, target=post_id)
    return _out(item, user.id)


@router.put("/posts/{post_id}", response_model=PostOut)
def edit_post(post_id: str, body: PostIn, user: CurrentUser = Depends(get_current_user)):
    post = _get_post(post_id)
    if post["author_id"] != user.id:
        raise HTTPException(403, "You can only edit your own posts")
    updated = table("forum").update_item(
        Key={"pk": post["pk"], "sk": "META"}, UpdateExpression="SET #t = :t, edited_at = :e",
        ExpressionAttributeNames={"#t": "text"}, ExpressionAttributeValues={":t": body.text, ":e": _now()},
        ReturnValues="ALL_NEW")["Attributes"]
    return _out(updated, user.id)


@router.delete("/posts/{post_id}", status_code=204)
def delete_post(post_id: str, bg: BackgroundTasks, user: CurrentUser = Depends(get_current_user)):
    post = _get_post(post_id)
    if post["author_id"] != user.id and not user.is_admin:
        raise HTTPException(403, "You can only delete your own posts")
    resp = table("forum").query(KeyConditionExpression=Key("pk").eq(post["pk"]))
    with table("forum").batch_writer() as batch:
        for it in resp.get("Items", []):
            batch.delete_item(Key={"pk": it["pk"], "sk": it["sk"]})
    bg.add_task(audit.record, "forum.delete", user.id, target=post_id,
                detail={"moderated": post["author_id"] != user.id})
    return Response(status_code=204)


@router.post("/posts/{post_id}/replies", response_model=ReplyOut, status_code=201,
             dependencies=[Depends(rate_limit("forum-reply", 40, 300))])
def reply(post_id: str, body: PostIn, user: CurrentUser = Depends(get_current_user)):
    post = _get_post(post_id)
    reply_id, now = uuid.uuid4().hex, _now()
    item = {"pk": post["pk"], "sk": f"REPLY#{now}#{reply_id}", "reply_id": reply_id, "text": body.text,
            "author_id": user.id, "author_name": display_name(user.id), "created_at": now}
    table("forum").put_item(Item=item)
    table("forum").update_item(Key={"pk": post["pk"], "sk": "META"}, UpdateExpression="ADD reply_count :one",
                               ExpressionAttributeValues={":one": 1})
    return ReplyOut(id=reply_id, text=body.text, author_id=user.id, author_name=item["author_name"], created_at=now)


@router.post("/posts/{post_id}/like", response_model=PostOut)
def like(post_id: str, user: CurrentUser = Depends(get_current_user)):
    """Idempotent: one like per user per post, enforced by a conditional write."""
    post = _get_post(post_id)
    try:
        table("forum").put_item(Item={"pk": post["pk"], "sk": f"LIKE#{user.id}", "created_at": _now()},
                                ConditionExpression="attribute_not_exists(sk)")
        table("forum").update_item(Key={"pk": post["pk"], "sk": "META"}, UpdateExpression="ADD likes :one",
                                   ExpressionAttributeValues={":one": 1})
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "ConditionalCheckFailedException":
            raise
    return _out(_get_post(post_id), user.id)


@router.delete("/posts/{post_id}/like", response_model=PostOut)
def unlike(post_id: str, user: CurrentUser = Depends(get_current_user)):
    post = _get_post(post_id)
    try:
        table("forum").delete_item(Key={"pk": post["pk"], "sk": f"LIKE#{user.id}"},
                                   ConditionExpression="attribute_exists(sk)")
        table("forum").update_item(Key={"pk": post["pk"], "sk": "META"}, UpdateExpression="ADD likes :neg",
                                   ExpressionAttributeValues={":neg": -1})
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "ConditionalCheckFailedException":
            raise
    return _out(_get_post(post_id), user.id)
