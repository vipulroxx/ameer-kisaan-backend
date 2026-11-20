import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from app.api import ai, auth, forum, inventory, photos, reports
from app.core.config import get_settings
from app.core.middleware import SecurityHeadersMiddleware, configure_logging
from app.db import dynamo

log = logging.getLogger("app")


def _warm_knowledge_base():
    try:
        from app.ai.rag import get_kb
        get_kb().ensure_index()
    except Exception as exc:  # assistant degrades gracefully; rest of the API is unaffected
        log.warning("knowledge index not built yet: %s", type(exc).__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    configure_logging(s.log_level)
    if s.app_env == "local":
        # docker-compose only: create DynamoDB Local tables and the MinIO bucket.
        from app.db.storage import ensure_bucket
        dynamo.create_tables()
        ensure_bucket()
    elif s.database_url and s.app_env != "test":
        dynamo.create_tables()  # Postgres: create the schema if missing (idempotent)
    if s.app_env != "test":
        threading.Thread(target=_warm_knowledge_base, daemon=True).start()
    yield


def create_app() -> FastAPI:
    s = get_settings()
    app = FastAPI(
        title=s.app_name, version="1.0.0", lifespan=lifespan,
        docs_url=None if s.is_prod else "/docs", redoc_url=None,
        openapi_url=None if s.is_prod else "/openapi.json",
    )
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(
        CORSMiddleware, allow_origins=s.cors_origins, allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        allow_headers=["Authorization", "Content-Type", "X-Requested-With", "X-Request-ID"],
        max_age=600,
    )

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError):
        # Echo field names and messages only, never the submitted values (could be passwords/PII).
        errors = [{"field": ".".join(str(p) for p in e["loc"][1:]), "message": e["msg"]} for e in exc.errors()]
        return JSONResponse({"detail": "Invalid input", "errors": errors}, status_code=422)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        log.exception("unhandled error", extra={"request_id": getattr(request.state, "request_id", None)})
        return JSONResponse({"detail": "Something went wrong. Please try again.",
                             "requestId": getattr(request.state, "request_id", None)}, status_code=500)

    @app.get("/health", include_in_schema=False)
    def health():
        return {"status": "ok"}

    @app.get("/health/ready", include_in_schema=False)
    def ready():
        try:
            dynamo.table("users").load()
            return {"status": "ready"}
        except Exception:
            return JSONResponse({"status": "unavailable"}, status_code=503)

    for r in (auth.router, inventory.router, forum.router, reports.router, ai.router, photos.router):
        app.include_router(r, prefix=s.api_prefix)
    if s.static_dir:
        _mount_web_app(app, Path(s.static_dir), s.api_prefix)
    return app


def _mount_web_app(app: FastAPI, root: Path, api_prefix: str) -> None:
    """Serve the built React app, falling back to index.html for client-side routes."""
    root = root.resolve()
    if not (root / "index.html").is_file():
        log.warning("STATIC_DIR %s has no index.html; web app not served", root)
        return
    api = api_prefix.strip("/") + "/"

    @app.api_route("/{path:path}", methods=["GET", "HEAD"], include_in_schema=False)
    def web(path: str):
        if path.startswith(api) or path == api.rstrip("/"):
            raise HTTPException(404, "Not Found")
        target = (root / path).resolve()
        if path and target.is_file() and target.is_relative_to(root):
            if path.startswith("static/"):
                cache = "public, max-age=31536000, immutable"  # content-hashed build assets
            elif path in ("service-worker.js", "index.html", "manifest.json"):
                cache = "no-cache"
            else:
                cache = "public, max-age=3600"
            return FileResponse(target, headers={"Cache-Control": cache})
        return FileResponse(root / "index.html", headers={"Cache-Control": "no-cache"})


app = create_app()
