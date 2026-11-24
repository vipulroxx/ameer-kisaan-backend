"""Admin console API: user paging, owner details on reports, immediate loss of admin rights."""
from conftest import make_user, register

from app.core.middleware import limiter


def test_user_list_pages_through_everyone_without_gaps_or_duplicates(client):
    admin, admin_id = make_user(client, "admin@example.com", admin=True)
    for i in range(7):
        limiter.reset()  # registration is rate limited per IP
        assert register(client, email=f"farmer{i}@example.com").status_code == 201
    seen, cursor, pages = [], None, 0
    while pages < 10:  # guard: a broken cursor must fail the test, not hang it
        # The cursor is "USER#<id>": it must be URL-encoded, or "#" starts a fragment and is dropped.
        params = {"limit": 3, **({"cursor": cursor} if cursor else {})}
        r = client.get("/api/v1/admin/users", params=params, headers=admin).json()
        assert r["items"], "a page must never come back empty while users remain"
        seen += [u["email"] for u in r["items"]]
        pages += 1
        cursor = r["nextCursor"]
        if not cursor:
            break
    assert not cursor, "paging never ended"
    assert sorted(seen) == sorted({*(f"farmer{i}@example.com" for i in range(7)), "admin@example.com"})
    assert len(seen) == len(set(seen)) == 8 and pages == 3


def test_user_list_shows_disabled_state(client):
    admin, _ = make_user(client, "admin@example.com", admin=True)
    _, farmer_id = make_user(client, "farmer@example.com")
    client.patch(f"/api/v1/admin/users/{farmer_id}", headers=admin, json={"disabled": True})
    users = {u["email"]: u for u in client.get("/api/v1/admin/users", headers=admin).json()["items"]}
    assert users["farmer@example.com"]["disabled"] is True and users["admin@example.com"]["disabled"] is False


def test_admin_report_list_includes_farmer_contact_but_farmers_never_see_it(client):
    farmer, _ = make_user(client, "farmer@example.com")
    admin, _ = make_user(client, "admin@example.com", admin=True)
    client.post("/api/v1/reports", headers=farmer, data={"crop": "Rice", "category": "pest",
                                                         "description": "Brown spots on leaves"})
    rep = client.get("/api/v1/reports/admin?status=open", headers=admin).json()[0]
    assert rep["ownerName"] == "Ramesh Kumar" and rep["ownerPhone"] == "9876543210"
    url = f"/api/v1/reports/{rep['ownerId']}/{rep['id']}"
    updated = client.patch(url, headers=admin, json={"status": "resolved", "adminNote": "Sprayed neem oil"}).json()
    assert updated["status"] == "resolved" and updated["ownerName"] == "Ramesh Kumar"
    mine = client.get("/api/v1/reports", headers=farmer).json()[0]
    assert mine["ownerName"] is None and mine["ownerPhone"] is None


def test_demoted_or_disabled_admin_loses_access_immediately(client):
    boss, _ = make_user(client, "boss@example.com", admin=True)
    other, other_id = make_user(client, "other@example.com", admin=True)
    assert client.get("/api/v1/admin/users", headers=other).status_code == 200
    client.patch(f"/api/v1/admin/users/{other_id}", headers=boss, json={"role": "user"})
    # Same, still-valid access token that says role=admin: refused at once.
    assert client.get("/api/v1/admin/users", headers=other).status_code == 403
    client.patch(f"/api/v1/admin/users/{other_id}", headers=boss, json={"role": "admin", "disabled": True})
    assert client.get("/api/v1/reports/admin", headers=other).status_code == 403


def test_admin_cannot_remove_own_admin_role(client):
    admin, admin_id = make_user(client, "admin@example.com", admin=True)
    assert client.patch(f"/api/v1/admin/users/{admin_id}", headers=admin, json={"role": "user"}).status_code == 400
