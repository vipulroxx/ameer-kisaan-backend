"""Inventory: each farmer sees and edits only their own items.

Ownership is enforced by the key design itself: every query uses the caller's
user id (from the verified token) as the partition key, so one user can never
address another user's items, even by guessing ids.
"""
import uuid
from datetime import datetime, timezone
from decimal import Decimal

from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Response

from app.core import audit
from app.core.deps import CurrentUser, get_current_user
from app.db.dynamo import table
from app.schemas import InventoryIn, InventoryOut

router = APIRouter(prefix="/inventory", tags=["inventory"])
MAX_ITEMS = 500


def _out(item: dict) -> InventoryOut:
    return InventoryOut(
        id=item["item_id"], product_name=item["product_name"], variety=item["variety"],
        quantity=float(item["quantity"]), price=float(item["price"]), harvest_date=item.get("harvest_date"),
        quality_grade=item.get("quality_grade", ""), packaging_type=item.get("packaging_type", ""),
        updated_at=item["updated_at"],
    )


def _to_item(owner: str, item_id: str, body: InventoryIn) -> dict:
    d = body.model_dump()
    return {
        "owner_id": owner, "item_id": item_id, "product_name": d["product_name"], "variety": d["variety"],
        "quantity": Decimal(str(d["quantity"])), "price": Decimal(str(d["price"])),
        "harvest_date": d["harvest_date"].isoformat() if d["harvest_date"] else None,
        "quality_grade": d["quality_grade"], "packaging_type": d["packaging_type"],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


@router.get("", response_model=list[InventoryOut])
def list_items(user: CurrentUser = Depends(get_current_user)):
    items, kwargs = [], {"KeyConditionExpression": Key("owner_id").eq(user.id)}
    while True:
        resp = table("inventory").query(**kwargs)
        items += resp.get("Items", [])
        if "LastEvaluatedKey" not in resp:
            break
        kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
    return [_out(i) for i in sorted(items, key=lambda i: i["updated_at"], reverse=True)]


@router.post("", response_model=InventoryOut, status_code=201)
def create_item(body: InventoryIn, bg: BackgroundTasks, user: CurrentUser = Depends(get_current_user)):
    count = table("inventory").query(KeyConditionExpression=Key("owner_id").eq(user.id), Select="COUNT")["Count"]
    if count >= MAX_ITEMS:
        raise HTTPException(400, f"Inventory limit of {MAX_ITEMS} items reached.")
    item = {k: v for k, v in _to_item(user.id, uuid.uuid4().hex, body).items() if v is not None}
    table("inventory").put_item(Item=item)
    bg.add_task(audit.record, "inventory.create", user.id, target=item["item_id"])
    return _out(item)


@router.put("/{item_id}", response_model=InventoryOut)
def update_item(item_id: str, body: InventoryIn, bg: BackgroundTasks, user: CurrentUser = Depends(get_current_user)):
    item = {k: v for k, v in _to_item(user.id, item_id, body).items() if v is not None}
    try:
        table("inventory").put_item(Item=item, ConditionExpression="attribute_exists(item_id)")
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "ConditionalCheckFailedException":
            raise HTTPException(404, "Item not found")
        raise
    bg.add_task(audit.record, "inventory.update", user.id, target=item_id)
    return _out(item)


@router.delete("/{item_id}", status_code=204)
def delete_item(item_id: str, bg: BackgroundTasks, user: CurrentUser = Depends(get_current_user)):
    try:
        table("inventory").delete_item(Key={"owner_id": user.id, "item_id": item_id},
                                       ConditionExpression="attribute_exists(item_id)")
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "ConditionalCheckFailedException":
            raise HTTPException(404, "Item not found")
        raise
    bg.add_task(audit.record, "inventory.delete", user.id, target=item_id)
    return Response(status_code=204)
