import time

import jwt

from app.db.dynamo import table
from tests.conftest import VALID, login, make_user, register

CSRF = {"X-Requested-With": "ameer-kisaan"}


def use_cookie(client, token):
    """Simulate a browser (or attacker) presenting a specific refresh cookie."""
    client.cookies.clear()
    client.cookies.set("ak_refresh", token)


def test_register_hashes_password_and_encrypts_identifiers(client):
    r = register(client)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["aadharId"] == "XXXX-XXXX-0172"
    assert body["panCardNumber"] == "BQXXXXXX1F"
    assert "password" not in body and "passwordHash" not in body

    stored = table("users").get_item(Key={"pk": f"USER#{body['id']}"})["Item"]
    assert stored["password_hash"].startswith("$argon2id$")
    assert VALID["password"] not in str(stored)
    assert "2345-6789-0172" not in str(stored) and "BQRPK4821F" not in str(stored)

    from app.core.security import get_cipher
    assert get_cipher().decrypt(stored["aadhar_enc"], f"aadhaar:{body['id']}") == "2345-6789-0172"


def test_register_rejects_role_escalation_and_duplicates(client):
    r = client.post("/api/v1/auth/register", json={**VALID, "role": "admin"})
    assert r.status_code == 422  # unknown fields are rejected outright
    assert register(client).status_code == 201
    assert register(client, email="RAMESH@Example.com ").status_code in (409, 422)
    assert register(client, email="ramesh@example.com").status_code == 409


def test_weak_password_rejected(client):
    for pw in ["short1", "allletterspassword", "password123", "ramesh12345678"]:
        r = register(client, password=pw)
        assert r.status_code == 422, pw


def test_validation_errors_do_not_echo_values(client):
    r = register(client, password="x", aadharId="not-an-aadhaar")
    assert r.status_code == 422
    assert "not-an-aadhaar" not in r.text


def test_login_issues_short_lived_access_token_and_httponly_cookie(client):
    register(client)
    r = login(client)
    assert r.status_code == 200
    claims = jwt.decode(r.json()["accessToken"], options={"verify_signature": False})
    assert claims["exp"] - claims["iat"] <= 15 * 60
    cookie = r.headers["set-cookie"]
    assert "ak_refresh=" in cookie and "HttpOnly" in cookie and "SameSite=strict" in cookie.replace("Strict", "strict")
    assert "Path=/api/v1/auth" in cookie


def test_wrong_password_and_unknown_email_look_the_same(client):
    register(client)
    a = login(client, password="WrongPassword99")
    b = login(client, email="nobody@example.com", password="WrongPassword99")
    assert a.status_code == b.status_code == 401
    assert a.json() == b.json()


def test_account_lockout_after_repeated_failures(client):
    register(client)
    for _ in range(5):
        assert login(client, password="WrongPassword99").status_code == 401
    assert login(client).status_code == 429  # even the right password is refused while locked


def test_protected_routes_require_valid_token(client, auth_headers):
    assert client.get("/api/v1/users/me").status_code == 401
    assert client.get("/api/v1/users/me", headers={"Authorization": "Bearer garbage"}).status_code == 401
    forged = jwt.encode({"sub": "x", "role": "admin", "type": "access", "exp": int(time.time()) + 60},
                        "not-the-secret", algorithm="HS256")
    assert client.get("/api/v1/users/me", headers={"Authorization": f"Bearer {forged}"}).status_code == 401
    none_alg = jwt.encode({"sub": "x", "role": "admin", "type": "access"}, key=None, algorithm="none")
    assert client.get("/api/v1/users/me", headers={"Authorization": f"Bearer {none_alg}"}).status_code == 401
    me = client.get("/api/v1/users/me", headers=auth_headers)
    assert me.status_code == 200 and me.json()["email"] == VALID["email"]


def test_refresh_rotation_and_reuse_detection(client):
    register(client)
    login(client)
    old = client.cookies.get("ak_refresh")

    assert client.post("/api/v1/auth/refresh").status_code == 403  # CSRF header required
    r = client.post("/api/v1/auth/refresh", headers=CSRF)
    assert r.status_code == 200 and r.json()["accessToken"]
    new = client.cookies.get("ak_refresh")
    assert new and new != old

    # Replaying the old (stolen) token revokes the whole session family...
    use_cookie(client, old)
    assert client.post("/api/v1/auth/refresh", headers=CSRF).status_code == 401
    # ...so the legitimate new token no longer works either.
    use_cookie(client, new)
    assert client.post("/api/v1/auth/refresh", headers=CSRF).status_code == 401


def test_logout_revokes_session(client):
    register(client)
    login(client)
    token = client.cookies.get("ak_refresh")
    assert client.post("/api/v1/auth/logout", headers=CSRF).status_code == 204
    use_cookie(client, token)
    assert client.post("/api/v1/auth/refresh", headers=CSRF).status_code == 401


def test_password_change_signs_out_other_sessions(client, auth_headers):
    other_device = client.cookies.get("ak_refresh")
    r = client.post("/api/v1/users/me/password", headers=auth_headers,
                    json={"currentPassword": VALID["password"], "newPassword": "NewHarvest2025x"})
    assert r.status_code == 200, r.text
    current = client.cookies.get("ak_refresh")
    use_cookie(client, other_device)
    assert client.post("/api/v1/auth/refresh", headers=CSRF).status_code == 401
    use_cookie(client, current)
    assert client.post("/api/v1/auth/refresh", headers=CSRF).status_code == 200
    assert login(client, password="NewHarvest2025x").status_code == 200


def test_profile_update(client, auth_headers):
    r = client.patch("/api/v1/users/me", headers=auth_headers, json={"preferredCrops": ["Wheat"], "farmId": "JH12345678"})
    assert r.status_code == 200 and r.json()["preferredCrops"] == ["Wheat"]
    assert client.patch("/api/v1/users/me", headers=auth_headers, json={"role": "admin"}).status_code == 422


def test_admin_rbac(client):
    farmer, farmer_id = make_user(client, "farmer@example.com")
    admin, _ = make_user(client, "admin@example.com", admin=True)
    assert client.get("/api/v1/admin/users", headers=farmer).status_code == 403
    r = client.get("/api/v1/admin/users", headers=admin)
    assert r.status_code == 200 and len(r.json()["items"]) == 2
    assert client.patch(f"/api/v1/admin/users/{farmer_id}", headers=admin, json={"disabled": True}).status_code == 200
    assert login(client, email="farmer@example.com").status_code == 401


def test_security_headers(client):
    r = client.get("/health")
    for h in ("x-content-type-options", "x-frame-options", "content-security-policy", "referrer-policy", "x-request-id"):
        assert h in r.headers


def test_audit_trail_written(client, auth_headers):
    from datetime import datetime, timezone
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    events = table("audit").query(KeyConditionExpression="#d = :d", ExpressionAttributeNames={"#d": "day"},
                                  ExpressionAttributeValues={":d": day})["Items"]
    actions = {e["action"] for e in events}
    assert {"user.register", "auth.login"} <= actions
    assert VALID["password"] not in str(events) and VALID["email"] not in str(events)


def test_login_rate_limit(client):
    for i in range(10):
        client.post("/api/v1/auth/login", json={"email": f"u{i}@example.com", "password": "Whatever123x"})
    assert client.post("/api/v1/auth/login", json={"email": "z@example.com", "password": "Whatever123x"}).status_code == 429


def test_spoofed_forwarded_for_cannot_bypass_rate_limit(client):
    for i in range(10):
        client.post("/api/v1/auth/login", json={"email": f"u{i}@example.com", "password": "Whatever123x"},
                    headers={"X-Forwarded-For": f"10.0.0.{i}, 203.0.113.7"})
    r = client.post("/api/v1/auth/login", json={"email": "z@example.com", "password": "Whatever123x"},
                    headers={"X-Forwarded-For": "1.2.3.4, 203.0.113.7"})
    assert r.status_code == 429
