import time
import uuid

from boto3.dynamodb.conditions import Key
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile

from app.ai import guardrails, rag, vision
from app.ai.providers import DailyBudgetExhausted, ProviderUnavailable
from app.api.reports import read_limited
from app.core import audit
from app.core.deps import CurrentUser, get_current_user, rate_limit
from app.db import storage
from app.db.dynamo import table
from app.schemas import AskRequest, AskResponse, Prediction, VisionResponse

router = APIRouter(tags=["ai"])
HISTORY_DAYS = 90


def _save_history(user_id: str, question: str, result: dict) -> None:
    now = time.time()
    redacted_q, _ = guardrails.redact_pii(question)
    table("chat").put_item(Item={
        "user_id": user_id, "message_id": f"{int(now * 1000):015d}#{uuid.uuid4().hex[:8]}",
        "question": redacted_q, "answer": result["answer"], "grounded": result["grounded"],
        "sources": [s["title"] for s in result["sources"]], "expires_at": int(now) + HISTORY_DAYS * 86400,
    })


@router.post("/assistant/ask", response_model=AskResponse,
             dependencies=[Depends(rate_limit("assistant", 20, 60))])
def ask(body: AskRequest, bg: BackgroundTasks, user: CurrentUser = Depends(get_current_user)):
    try:
        result = rag.answer(body.question, body.language)
    except guardrails.GuardrailViolation as exc:
        bg.add_task(audit.record, "assistant.blocked", user.id, outcome="blocked")
        raise HTTPException(400, str(exc))
    except DailyBudgetExhausted:
        # Zero-cost limit (see Settings.ai_daily_*): resets at 00:00 UTC, 5:30 AM in India.
        raise HTTPException(503, "Kisaan AI has answered all the questions it can for free today. "
                                 "It will be back tomorrow after 5:30 AM.", headers={"Retry-After": "3600"})
    except ProviderUnavailable:
        raise HTTPException(503, "The assistant is temporarily unavailable. Please try again shortly.")
    bg.add_task(_save_history, user.id, body.question, result)
    return result


@router.get("/assistant/history")
def history(limit: int = 20, user: CurrentUser = Depends(get_current_user)):
    resp = table("chat").query(KeyConditionExpression=Key("user_id").eq(user.id), ScanIndexForward=False,
                               Limit=min(max(limit, 1), 50))
    return {"items": [{"question": i["question"], "answer": i["answer"], "grounded": i["grounded"],
                       "sources": i.get("sources", [])} for i in resp.get("Items", [])]}


@router.post("/vision/identify-crop", response_model=VisionResponse,
             dependencies=[Depends(rate_limit("vision", 20, 60))])
async def identify_crop(photo: UploadFile = File(...), user: CurrentUser = Depends(get_current_user)):
    try:
        clean, _ = storage.sanitize_image(await read_limited(photo))
    except storage.InvalidImage as exc:
        raise HTTPException(415, str(exc))
    try:
        clf = vision.get_classifier()
    except vision.ModelNotReady as exc:
        raise HTTPException(503, str(exc))
    preds = clf.predict(clean)
    return VisionResponse(predictions=[Prediction(label=label, probability=round(p, 4)) for label, p in preds],
                          model_version=clf.version)
