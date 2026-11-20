"""Single-service hosting: signed photo links, serving the web app, split AI providers."""
import io
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.core.config import get_settings
from app.db import storage


def jpeg() -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (32, 32), (40, 120, 40)).save(out, format="JPEG")
    return out.getvalue()


@pytest.fixture
def proxy_mode(monkeypatch):
    monkeypatch.setattr(get_settings(), "photo_url_mode", "proxy")


def _report(client, auth_headers):
    r = client.post("/api/v1/reports", headers=auth_headers,
                    data={"crop": "Tomato", "category": "pest", "description": "Leaves curling and yellow"},
                    files=[("photos", ("leaf.jpg", jpeg(), "image/jpeg"))])
    assert r.status_code == 201, r.text
    return r.json()["photoUrls"][0]


def test_proxy_photo_link_serves_the_private_photo(client, auth_headers, proxy_mode):
    url = _report(client, auth_headers)
    assert url.startswith("/api/v1/photos/reports/") and "sig=" in url and "X-Amz-Signature" not in url
    r = client.get(url)  # an <img> tag sends no Authorization header
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    Image.open(io.BytesIO(r.content)).verify()
    assert r.headers["cache-control"] == "no-store"


def test_proxy_photo_link_rejects_tampering_and_expiry(client, auth_headers, proxy_mode):
    url = _report(client, auth_headers)
    path, query = url.split("?")
    params = dict(p.split("=") for p in query.split("&"))
    bad_sig = f"{path}?exp={params['exp']}&sig={'0' * 64}"
    assert client.get(bad_sig).status_code == 403
    later = int(params["exp"]) + 3600  # extending the expiry invalidates the signature
    assert client.get(f"{path}?exp={later}&sig={params['sig']}").status_code == 403
    other = path.rsplit("/", 1)[0] + "/" + "f" * 32 + ".jpg"  # a signature is bound to one object key
    assert client.get(f"{other}?exp={params['exp']}&sig={params['sig']}").status_code == 403
    expired = int(time.time()) - 1
    key = path.removeprefix("/api/v1/photos/")
    assert client.get(f"{path}?exp={expired}&sig={storage.photo_signature(key, expired)}").status_code == 403
    assert client.get("/api/v1/photos/../../etc/passwd?exp=9999999999&sig=x").status_code in (403, 404)


def test_proxy_link_for_missing_object_is_404(client, proxy_mode):
    key = "reports/u1/r1/" + "a" * 32 + ".jpg"
    exp = int(time.time()) + 60
    assert client.get(f"/api/v1/photos/{key}?exp={exp}&sig={storage.photo_signature(key, exp)}").status_code == 404


@pytest.fixture
def web_client(tmp_path, monkeypatch):
    (tmp_path / "static" / "js").mkdir(parents=True)
    (tmp_path / "videos").mkdir()
    (tmp_path / "index.html").write_text("<!doctype html><title>Ameer Kisaan</title><div id=root></div>")
    (tmp_path / "static" / "js" / "main.abc123.js").write_text("console.log('app')")
    (tmp_path / "videos" / "demo.mp4").write_bytes(bytes(range(256)) * 40)
    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path.parent / "secret.txt").write_text("do not serve")
    monkeypatch.setattr(get_settings(), "static_dir", str(tmp_path))
    from app.main import create_app
    with TestClient(create_app()) as c:
        yield c


def test_web_app_is_served_with_spa_fallback(web_client):
    for path in ("/", "/login", "/app/weather", "/app/forum"):
        r = web_client.get(path)
        assert r.status_code == 200 and "Ameer Kisaan" in r.text, path
        assert r.headers["cache-control"] == "no-cache"
        csp = r.headers["content-security-policy"]
        assert "script-src 'self'" in csp and "frame-ancestors 'none'" in csp and "media-src 'self'" in csp
        assert r.headers["x-frame-options"] == "DENY"
    asset = web_client.get("/static/js/main.abc123.js")
    assert asset.status_code == 200 and "immutable" in asset.headers["cache-control"]
    assert web_client.get("/manifest.json").headers["cache-control"] == "no-cache"
    assert web_client.head("/").status_code == 200


def test_videos_support_range_requests(web_client):
    r = web_client.get("/videos/demo.mp4", headers={"Range": "bytes=0-99"})
    assert r.status_code == 206 and len(r.content) == 100
    assert r.headers["content-type"] == "video/mp4"


def test_web_app_never_serves_files_outside_its_folder(web_client):
    for path in ("/../secret.txt", "/%2e%2e/secret.txt", "/static/../../secret.txt"):
        r = web_client.get(path)
        assert "do not serve" not in r.text, path


def test_api_routes_and_headers_are_unchanged_when_web_app_is_served(web_client):
    r = web_client.get("/api/v1/does-not-exist")
    assert r.status_code == 404 and r.json() == {"detail": "Not Found"}
    assert r.headers["content-security-policy"] == "default-src 'none'; frame-ancestors 'none'"
    assert web_client.get("/health").json() == {"status": "ok"}
    assert web_client.get("/api/v1/users/me").status_code == 401  # real API route still wins over the catch-all


def test_embeddings_can_use_a_separate_provider(monkeypatch):
    from app.ai import providers
    s = get_settings()
    monkeypatch.setattr(s, "openai_api_key", "chat-key")
    monkeypatch.setattr(s, "openai_base_url", "https://chat.example/v1")
    monkeypatch.setattr(s, "openai_embed_base_url", "https://embed.example/v1/")
    monkeypatch.setattr(s, "openai_embed_api_key", "embed-key")
    calls = []

    def fake_post(url, payload, headers=None, timeout=60.0):
        calls.append((url, headers["Authorization"]))
        if url.endswith("/embeddings"):
            return {"data": [{"index": 1, "embedding": [0.0, 1.0]}, {"index": 0, "embedding": [1.0, 0.0]}]}
        return {"choices": [{"message": {"content": "ok [1]"}}]}

    monkeypatch.setattr(providers, "_post", fake_post)
    assert providers.OpenAIEmbedder(s).embed(["a", "b"]) == [[1.0, 0.0], [0.0, 1.0]]
    assert providers.OpenAILLM(s).chat("sys", "q") == "ok [1]"
    assert calls == [("https://embed.example/v1/embeddings", "Bearer embed-key"),
                     ("https://chat.example/v1/chat/completions", "Bearer chat-key")]


def test_embeddings_default_to_the_chat_provider(monkeypatch):
    from app.ai import providers
    s = get_settings()
    monkeypatch.setattr(s, "openai_api_key", "one-key")
    monkeypatch.setattr(s, "openai_embed_base_url", None)
    monkeypatch.setattr(s, "openai_embed_api_key", None)
    e = providers.OpenAIEmbedder(s)
    assert (e.base_url, e.api_key) == (s.openai_base_url.rstrip("/"), "one-key")


def test_client_ip_respects_proxy_configuration(monkeypatch):
    from starlette.requests import Request

    from app.core.middleware import client_ip

    def req(headers):
        return Request({"type": "http", "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
                        "client": ("10.0.0.9", 1234)})

    s = get_settings()
    spoofed = {"X-Forwarded-For": "6.6.6.6, 203.0.113.7, 10.1.1.1"}
    assert client_ip(req(spoofed)) == "10.1.1.1"  # default: one trusted proxy
    monkeypatch.setattr(s, "trusted_proxy_hops", 2)
    assert client_ip(req(spoofed)) == "203.0.113.7"
    monkeypatch.setattr(s, "trusted_proxy_hops", 9)
    assert client_ip(req(spoofed)) == "6.6.6.6"  # never past the leftmost entry
    monkeypatch.setattr(s, "client_ip_header", "cf-connecting-ip")
    assert client_ip(req({**spoofed, "CF-Connecting-IP": "198.51.100.4"})) == "198.51.100.4"
    assert client_ip(req({})) == "10.0.0.9"
