"""Promote an existing account to admin. Admins can never be created through the public API.

    docker compose exec backend python scripts/create_admin.py you@example.com
"""
import sys

from app.services import users


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: create_admin.py <email>")
    user = users.get_user_by_email(sys.argv[1])
    if not user:
        sys.exit("No account with that email. Register in the app first.")
    users.update_profile(user["id"], {"role": "admin"})
    users.set_sessions_cutoff(user["id"])  # log in again to receive an admin token
    print(f"{user['email']} is now an admin. Log in again to use admin features.")


if __name__ == "__main__":
    main()
