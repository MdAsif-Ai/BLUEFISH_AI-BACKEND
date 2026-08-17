"""
BlueFish AI — Production Auth E2E Test
======================================
Tests the production Supabase auth flows:
  1. POST /api/v1/auth/register → Creates user in Supabase Auth & profiles table
  2. POST /api/v1/auth/login    → Returns real Supabase JWT access token
  3. GET  /api/v1/auth/me       → Verifies JWT against Supabase & gets profile row
  4. POST /api/v1/auth/refresh  → Refreshes Supabase session
  5. POST /api/v1/auth/logout   → Signs out Supabase session
"""

import json
import sys
import uuid
import requests

BASE_URL = "http://127.0.0.1:8000"

def test_production_auth_flow():
    print("\n" + "=" * 60)
    print("  BlueFish AI — Production Auth E2E Test Suite")
    print("=" * 60 + "\n")

    # 1. Generate unique user details
    uid = str(uuid.uuid4())[:8]
    email = f"fisherman_{uid}@gmail.com"
    password = "SecurePassword123!"
    full_name = f"Tamil Fisherman {uid}"
    role = "fisherman"

    print(f"[*] Registering new user via Supabase Auth...")
    print(f"    Email: {email}")

    # 2. Register
    reg_res = requests.post(
        f"{BASE_URL}/api/v1/auth/register",
        json={
            "email": email,
            "password": password,
            "full_name": full_name,
            "role": role,
            "phone": "+919876543210",
            "preferred_language": "ta",
        },
        timeout=15,
    )

    print(f"    Response Code: {reg_res.status_code}")
    print(f"    Response Body: {reg_res.text}")

    assert reg_res.status_code == 201, f"Registration failed: {reg_res.text}"
    reg_data = reg_res.json()
    user_id = reg_data.get("user_id")
    print(f" ✅ User successfully created in Supabase Auth & profiles table!")
    print(f"    User ID: {user_id}")

    # 3. Login
    print(f"\n[*] Logging in with Supabase Auth...")
    login_res = requests.post(
        f"{BASE_URL}/api/v1/auth/login",
        json={"email": email, "password": password},
        timeout=15,
    )

    print(f"    Response Code: {login_res.status_code}")
    assert login_res.status_code == 200, f"Login failed: {login_res.text}"
    login_data = login_res.json()
    token = login_data["access_token"]
    refresh_tok = login_data["refresh_token"]
    print(f" ✅ Login successful! JWT Access Token received.")
    print(f"    Token snippet: {token[:30]}...")

    # 4. /me Profile retrieval
    print(f"\n[*] Fetching /me authenticated profile...")
    me_res = requests.get(
        f"{BASE_URL}/api/v1/auth/me",
        headers={"Authorization": f"Bearer {token}"},
        timeout=15,
    )
    print(f"    Response Code: {me_res.status_code}")
    print(f"    Response Body: {me_res.text}")
    assert me_res.status_code == 200, f"Fetch /me failed: {me_res.text}"
    me_data = me_res.json()
    assert me_data["user_id"] == user_id
    assert me_data["email"] == email
    print(f" ✅ /me returned verified Supabase profile successfully!")

    # 5. Refresh token
    print(f"\n[*] Testing token refresh...")
    ref_res = requests.post(
        f"{BASE_URL}/api/v1/auth/refresh",
        json={"refresh_token": refresh_tok},
        timeout=15,
    )
    print(f"    Response Code: {ref_res.status_code}")
    assert ref_res.status_code == 200, f"Token refresh failed: {ref_res.text}"
    print(f" ✅ Session refresh successful!")

    # 6. Logout
    print(f"\n[*] Testing logout...")
    out_res = requests.post(
        f"{BASE_URL}/api/v1/auth/logout",
        headers={"Authorization": f"Bearer {token}"},
        timeout=15,
    )
    print(f"    Response Code: {out_res.status_code}")
    assert out_res.status_code == 200
    print(f" ✅ Logout successful!")

    print("\n" + "=" * 60)
    print(" 🎉 ALL PRODUCTION AUTH TESTS PASSED PERFECTLY!")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    try:
        test_production_auth_flow()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        sys.exit(1)
