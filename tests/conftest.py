import base64
import os

os.environ.update({
    "APP_ENV": "test",
    "JWT_SECRET": "test-secret-" + "x" * 40,
    "FIELD_ENCRYPTION_KEY": base64.b64encode(b"k" * 32).decode(),
    "COOKIE_SECURE": "false",
    "LLM_PROVIDER": "fake",
    "EMBEDDINGS_PROVIDER": "fake",
    "INDEX_CACHE_DIR": "",
    "RAG_MAX_DISTANCE": "0.95",
    "AWS_ACCESS_KEY_ID": "testing",
    "AWS_SECRET_ACCESS_KEY": "testing",
    "AWS_DEFAULT_REGION": "ap-south-1",
    "KNOWLEDGE_DIR": os.path.join(os.path.dirname(__file__), "..", "knowledge"),
    "VISION_MODEL_PATH": "/nonexistent/model.tflite",
})
# Set TEST_DATABASE_URL (a throwaway Postgres) to run the whole suite against the
# Postgres table backend (app/db/pgtable.py) instead of moto's DynamoDB.
PG_URL = os.getenv("TEST_DATABASE_URL")
if PG_URL:
    os.environ["DATABASE_URL"] = PG_URL

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from moto import mock_aws  # noqa: E402

from app.core.middleware import limiter  # noqa: E402
from app.db import dynamo, storage  # noqa: E402


@pytest.fixture(autouse=True)
def aws():
    with mock_aws():
        dynamo.dynamodb.cache_clear()
        storage._s3.cache_clear()
        dynamo.create_tables()
        if PG_URL:
            with dynamo.dynamodb().backend.pool.connection() as conn:
                conn.execute("TRUNCATE ak_items, ak_blobs")
        storage.ensure_bucket()
        limiter.reset()
        yield


@pytest.fixture
def client():
    from app.main import app
    with TestClient(app) as c:
        yield c


VALID = {
    "username": "Ramesh Kumar", "email": "ramesh@example.com", "phoneNumber": "9876543210",
    "password": "Kheti2024Ranchi", "aadharId": "2345-6789-0172", "panCardNumber": "BQRPK4821F",
    "farmId": "FARM239655", "preferredCrops": ["Rice", "Maize"], "userType": "Farmer",
}


def register(client, **overrides):
    body = {**VALID, **overrides}
    return client.post("/api/v1/auth/register", json=body)


def login(client, email=VALID["email"], password=VALID["password"]):
    return client.post("/api/v1/auth/login", json={"email": email, "password": password})


@pytest.fixture
def auth_headers(client):
    assert register(client).status_code == 201
    token = login(client).json()["accessToken"]
    return {"Authorization": f"Bearer {token}"}


def make_user(client, email, admin=False):
    register(client, email=email)
    if admin:
        from app.services import users
        u = users.get_user_by_email(email)
        users.update_profile(u["id"], {"role": "admin"})
    r = login(client, email=email)
    return {"Authorization": f"Bearer {r.json()['accessToken']}"}, r.json()["user"]["id"]
