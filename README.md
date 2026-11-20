# ameer-kisaan-backend

FastAPI service for Ameer Kisaan: authentication, user profiles, inventory, forum, crop issue reports, the Kisaan AI assistant and the crop image classifier. Data lives in DynamoDB, photos in S3.

## Develop

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
pytest            # 35 tests; AWS is mocked with moto, AI uses offline fake providers
ruff check .      # includes the bandit (S) security rules
pip-audit -r requirements.txt
```

To run the API on its own, use the repo-root `docker-compose.yml`, or point it at any DynamoDB/S3 endpoint with the variables in `.env.example`.

## Layout

```
app/
  main.py              app factory, middleware, error handlers, health checks
  core/                config, security (hashing, JWT, field encryption), deps (auth/RBAC/rate limits),
                       middleware (headers, request ids, JSON logs), audit trail
  db/                  dynamo.py (table definitions), storage.py (S3 + image sanitising)
  services/users.py    accounts, login lockout, refresh-token rotation
  api/                 auth, inventory, forum, reports, ai routers
  ai/                  providers (Ollama / OpenAI / Azure / fake), rag.py, guardrails.py, vision.py
knowledge/             24 markdown docs generated from the app's crop, pest, risk and scheme content
scripts/               create_admin.py, train_crop_classifier.py
models/                trained classifier goes here (git-ignored)
```

## API (`/api/v1`)

| Method | Path | Auth | Purpose |
|---|---|---|---|
| POST | `/auth/register` | - | Create account (role is always `user`) |
| POST | `/auth/login` | - | Returns access token; sets refresh cookie |
| POST | `/auth/refresh` | cookie + `X-Requested-With` | Rotate refresh token, new access token |
| POST | `/auth/logout` | cookie + `X-Requested-With` | Revoke session |
| GET/PATCH | `/users/me` | user | Profile (identifiers masked) |
| POST | `/users/me/password` | user | Change password, sign out other devices |
| GET/POST | `/inventory` | user | List / add own items |
| PUT/DELETE | `/inventory/{id}` | user | Edit / delete own item |
| GET/POST | `/forum/posts` | user | Paginated feed / new post |
| PUT/DELETE | `/forum/posts/{id}` | author (delete: author or admin) | Edit / delete |
| POST | `/forum/posts/{id}/replies` | user | Reply |
| POST/DELETE | `/forum/posts/{id}/like` | user | Like / unlike (idempotent) |
| GET/POST | `/reports` | user | Own reports / new report with up to 4 photos (multipart) |
| GET | `/reports/admin?status=open` | admin | Review queue |
| PATCH | `/reports/{owner}/{id}` | admin | Set status + officer note |
| POST | `/assistant/ask` | user | Grounded answer + sources (`en` or `hi`) |
| GET | `/assistant/history` | user | Last questions and answers (PII redacted, 90-day TTL) |
| POST | `/vision/identify-crop` | user | Top-3 crop predictions (503 until a model is trained) |
| GET | `/admin/users`, PATCH `/admin/users/{id}` | admin | Manage roles / disable accounts |
| GET | `/admin/audit?day=YYYY-MM-DD` | admin | Audit events |

Interactive docs at `/docs` in local/test environments (disabled in staging/prod).

## Kisaan AI

1. The question passes guardrails (prompt-injection patterns, control characters) and has Aadhaar/PAN/phone/email redacted.
2. Crop names (including Hindi ones such as *gehu*, *dhan*, *makka*) are detected; passages tagged with those crops are retrieved first, then general passages fill the remaining slots (cosine similarity over embeddings, in process).
3. If nothing passes the relevance threshold, the assistant says so without calling the LLM.
4. Otherwise the LLM gets only the retrieved passages, with instructions to cite them and treat them as data. The answer is PII-checked again and returned with its sources.

To add knowledge, drop a markdown file in `knowledge/` with front matter like:

```
---
topic: crop-guide
crops: Wheat, Rice
---
# Title
## Section per crop or topic
```

The index rebuilds automatically on the next start.
