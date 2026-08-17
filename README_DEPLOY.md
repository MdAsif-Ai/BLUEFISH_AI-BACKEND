BlueFish AI Backend — Deploy to Render (step-by-step)

1) Prerequisites
   - GitHub repo containing this project (this workspace).
   - Render account.
   - Supabase project credentials: SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, SUPABASE_JWT_SECRET.
   - Redis instance URL (Redis Cloud or Render Redis): REDIS_URL.

2) Create Web Service on Render (UI)
   - New → Web Service → Connect GitHub → select this repo and branch `main`.
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1`
   - Instance: choose 8GB RAM plan.
   - Health Check Path: `/health`

3) Add Environment Variables (Service → Environment)
   - ENVIRONMENT = production
   - SUPABASE_URL = <your_supabase_url>
   - SUPABASE_SERVICE_ROLE_KEY = <your_service_role_key>
   - SUPABASE_JWT_SECRET = <your_jwt_secret>
   - REDIS_URL = <your_redis_url>
   - CELERY_BROKER_URL = <redis_broker_url> (optional)
   - CELERY_RESULT_BACKEND = <redis_result_backend> (optional)
   - ML_MODELS_BUCKET = ml-models

4) Create Background Services (Render)
   - New → Background Worker → same repo/branch.
   - Worker start command: `celery -A celery_worker.celery_app worker --loglevel=info`
   - Add same env vars.
   - Repeat to create Beat: `celery -A celery_worker.celery_app beat --loglevel=info`

5) Verify model loading
   - Open Web Service logs in Render after deployment.
   - Search for `bluefish.model_loader` messages showing downloads and `✓ Model X loaded.`
   - If you see OOM, upgrade instance size or consider lazy-loading models.

6) Smoke Test (Postman / curl)
   - Register:
     ```bash
     curl -X POST https://<your-render-host>/api/v1/auth/register \
       -H "Content-Type: application/json" \
       -d '{"email":"qa+1@example.com","password":"SecurePassword123!","full_name":"QA","role":"fisherman"}'
     ```
   - Login:
     ```bash
     curl -X POST https://<your-render-host>/api/v1/auth/login \
       -H "Content-Type: application/json" \
       -d '{"email":"qa+1@example.com","password":"SecurePassword123!"}'
     ```
   - /me:
     ```bash
     curl -H "Authorization: Bearer <access_token>" https://<your-render-host>/api/v1/auth/me
     ```

7) Troubleshooting
   - Redis auth errors: verify `REDIS_URL` credentials.
   - Supabase 404 on model downloads: ensure `ML_MODELS_BUCKET` contains the objects and `SUPABASE_SERVICE_ROLE_KEY` has storage access.
   - Rate limits (429): check Supabase project quota; use a test project for heavy load.

