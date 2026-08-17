#!/usr/bin/env bash
# Usage: RENDER_HOST=https://your-service.onrender.com ./smoke_test_prod.sh
set -euo pipefail
HOST=${RENDER_HOST:-}
if [ -z "$HOST" ]; then
  echo "Set RENDER_HOST environment variable to your Render service URL"
  exit 1
fi
EMAIL="qa+$(date +%s)@example.com"
PASSWORD="SecurePassword123!"

echo "Registering $EMAIL..."
REG=$(curl -s -w "\n%{http_code}" -X POST "$HOST/api/v1/auth/register" -H "Content-Type: application/json" -d "{\"email\":\"$EMAIL\",\"password\":\"$PASSWORD\",\"full_name\":\"QA\",\"role\":\"fisherman\"}")
printf "Register response:\n%s\n" "$REG"

echo "Logging in..."
LOGIN=$(curl -s -X POST "$HOST/api/v1/auth/login" -H "Content-Type: application/json" -d "{\"email\":\"$EMAIL\",\"password\":\"$PASSWORD\"}")
echo "$LOGIN"

TOKEN=$(echo "$LOGIN" | python -c "import sys, json; print(json.load(sys.stdin).get('access_token'))")
if [ -z "$TOKEN" ] || [ "$TOKEN" == "None" ]; then
  echo "Login failed or no access token returned"
  exit 2
fi

echo "Fetching /me with token..."
curl -s -H "Authorization: Bearer $TOKEN" "$HOST/api/v1/auth/me" | jq || true

echo "Smoke test completed."
