import io

from PIL import Image

from tests.conftest import make_user

ITEM = {"productName": "Rice", "variety": "Basmati", "quantity": 1000, "price": 1800, "harvestDate": "2025-10-01",
        "qualityGrade": "A", "packagingType": "Bag"}


def jpeg_with_gps() -> bytes:
    img = Image.new("RGB", (64, 64), (40, 160, 60))
    exif = Image.Exif()
    exif[0x8825] = {2: (23.0, 21.0, 0.0), 4: (85.0, 19.0, 0.0)}  # GPSInfo
    exif[0x010F] = "PhoneMaker"
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    return buf.getvalue()


def test_inventory_crud_and_isolation(client):
    alice, _ = make_user(client, "alice@example.com")
    bob, _ = make_user(client, "bob@example.com")

    r = client.post("/api/v1/inventory", headers=alice, json=ITEM)
    assert r.status_code == 201, r.text
    item_id = r.json()["id"]
    assert client.get("/api/v1/inventory", headers=alice).json()[0]["productName"] == "Rice"

    # Bob can neither see nor modify Alice's item
    assert client.get("/api/v1/inventory", headers=bob).json() == []
    assert client.put(f"/api/v1/inventory/{item_id}", headers=bob, json=ITEM).status_code == 404
    assert client.delete(f"/api/v1/inventory/{item_id}", headers=bob).status_code == 404

    r = client.put(f"/api/v1/inventory/{item_id}", headers=alice, json={**ITEM, "quantity": 750})
    assert r.json()["quantity"] == 750
    assert client.delete(f"/api/v1/inventory/{item_id}", headers=alice).status_code == 204
    assert client.get("/api/v1/inventory", headers=alice).json() == []


def test_inventory_validation(client):
    alice, _ = make_user(client, "alice@example.com")
    assert client.post("/api/v1/inventory", headers=alice, json={**ITEM, "quantity": -5}).status_code == 422
    assert client.post("/api/v1/inventory", headers=alice, json={**ITEM, "productName": "x" * 200}).status_code == 422
    assert client.post("/api/v1/inventory", headers=alice, json={**ITEM, "ownerId": "someone"}).status_code == 422


def test_forum_flow(client):
    alice, _ = make_user(client, "alice@example.com")
    bob, _ = make_user(client, "bob@example.com")
    post = client.post("/api/v1/forum/posts", headers=alice, json={"text": "Best time to sow wheat in Ranchi?"}).json()
    assert post["authorName"] == "Ramesh Kumar"

    client.post(f"/api/v1/forum/posts/{post['id']}/replies", headers=bob, json={"text": "Mid November."})
    for _ in range(3):  # likes are idempotent
        liked = client.post(f"/api/v1/forum/posts/{post['id']}/like", headers=bob).json()
    assert liked["likes"] == 1 and liked["likedByMe"]

    feed = client.get("/api/v1/forum/posts", headers=bob).json()["items"]
    assert feed[0]["replyCount"] == 1 and feed[0]["replies"][0]["text"] == "Mid November."

    # Only the author may edit/delete
    assert client.put(f"/api/v1/forum/posts/{post['id']}", headers=bob, json={"text": "hacked"}).status_code == 403
    assert client.delete(f"/api/v1/forum/posts/{post['id']}", headers=bob).status_code == 403
    assert client.put(f"/api/v1/forum/posts/{post['id']}", headers=alice, json={"text": "Edited"}).json()["editedAt"]
    assert client.delete(f"/api/v1/forum/posts/{post['id']}", headers=alice).status_code == 204
    assert client.get("/api/v1/forum/posts", headers=bob).json()["items"] == []


def test_forum_pagination_and_bad_cursor(client):
    alice, _ = make_user(client, "alice@example.com")
    for i in range(5):
        client.post("/api/v1/forum/posts", headers=alice, json={"text": f"post {i}"})
    page = client.get("/api/v1/forum/posts?limit=2", headers=alice).json()
    assert len(page["items"]) == 2 and page["nextCursor"]
    page2 = client.get(f"/api/v1/forum/posts?limit=2&cursor={page['nextCursor']}", headers=alice).json()
    assert {p["id"] for p in page2["items"]}.isdisjoint({p["id"] for p in page["items"]})
    assert client.get("/api/v1/forum/posts?cursor=%7Bjunk", headers=alice).status_code == 400


def test_admin_can_moderate_forum(client):
    alice, _ = make_user(client, "alice@example.com")
    admin, _ = make_user(client, "admin@example.com", admin=True)
    post = client.post("/api/v1/forum/posts", headers=alice, json={"text": "spam"}).json()
    assert client.delete(f"/api/v1/forum/posts/{post['id']}", headers=admin).status_code == 204


def test_report_upload_strips_exif_and_is_private(client):
    alice, alice_id = make_user(client, "alice@example.com")
    bob, _ = make_user(client, "bob@example.com")
    r = client.post("/api/v1/reports", headers=alice,
                    data={"crop": "Rice", "category": "pest", "severity": "high",
                          "description": "Brown spots spreading on leaves", "district": "Ranchi"},
                    files=[("photos", ("leaf.jpg", jpeg_with_gps(), "image/jpeg"))])
    assert r.status_code == 201, r.text
    rep = r.json()
    assert rep["status"] == "open" and len(rep["photoUrls"]) == 1
    assert "X-Amz-Signature" in rep["photoUrls"][0]  # short-lived presigned URL, bucket is private

    from app.core.config import get_settings
    from app.db.storage import _s3
    key = _s3().list_objects_v2(Bucket=get_settings().s3_bucket)["Contents"][0]["Key"]
    assert key.startswith(f"reports/{alice_id}/")
    stored = _s3().get_object(Bucket=get_settings().s3_bucket, Key=key)["Body"].read()
    assert not Image.open(io.BytesIO(stored)).getexif()  # GPS + device metadata removed

    assert client.get("/api/v1/reports", headers=bob).json() == []


def test_report_rejects_non_images_and_oversize(client):
    alice, _ = make_user(client, "alice@example.com")
    base = {"crop": "Rice", "category": "pest", "description": "Something is wrong"}
    fake = client.post("/api/v1/reports", headers=alice, data=base,
                       files=[("photos", ("x.jpg", b"<?php system($_GET['c']); ?>", "image/jpeg"))])
    assert fake.status_code == 415
    big = client.post("/api/v1/reports", headers=alice, data=base,
                      files=[("photos", ("big.jpg", b"\xff" * (6 * 1024 * 1024), "image/jpeg"))])
    assert big.status_code == 413
    assert client.post("/api/v1/reports", headers=alice, data={**base, "category": "nonsense"}).status_code == 422


def test_report_status_admin_only(client):
    alice, alice_id = make_user(client, "alice@example.com")
    admin, _ = make_user(client, "admin@example.com", admin=True)
    rep = client.post("/api/v1/reports", headers=alice,
                      data={"crop": "Maize", "category": "weather", "description": "Hail damage last night"}).json()
    url = f"/api/v1/reports/{alice_id}/{rep['id']}"
    assert client.patch(url, headers=alice, json={"status": "resolved"}).status_code == 403
    assert client.get("/api/v1/reports/admin", headers=alice).status_code == 403
    assert len(client.get("/api/v1/reports/admin?status=open", headers=admin).json()) == 1
    r = client.patch(url, headers=admin, json={"status": "in_review", "adminNote": "Officer visiting Tuesday"})
    assert r.status_code == 200 and r.json()["status"] == "in_review"
    assert client.get("/api/v1/reports", headers=alice).json()[0]["adminNote"] == "Officer visiting Tuesday"
