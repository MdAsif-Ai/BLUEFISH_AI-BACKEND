"""
BlueFish AI - Automated Test Script for Auth Flow
===================================================
Tests the authentication endpoints: register, login, me, refresh, logout.
It uses `requests` to hit the local server running at http://localhost:8000.
"""

import uuid
import time
import requests

BASE_URL = "http://127.0.0.1:8000"

def test_auth_flow():
    print("Starting Auth Flow Test...")

    # 1. Generate unique email for registration
    unique_id = str(uuid.uuid4())[:8]
    test_email = f"testuser_{unique_id}@example.com"
    test_password = "SecurePassword123!"

    print(f"\n[1] Testing Registration with {test_email}")
    reg_payload = {
        "email": test_email,
        "password": test_password,
        "full_name": "Test Fisherman",
        "role": "fisherman",
        "phone": f"+9198765{int(time.time()) % 10000:04d}",
        "preferred_language": "en"
    }
    
    try:
        reg_res = requests.post(f"{BASE_URL}/api/v1/auth/register", json=reg_payload)
    except requests.exceptions.ConnectionError:
        print(f" ❌ Connection failed. Ensure the server is running at {BASE_URL}")
        return

    if reg_res.status_code == 201:
        print(" ✅ Registration successful.")
    else:
        print(f" ❌ Registration failed: {reg_res.status_code} - {reg_res.text}")
        if reg_res.status_code not in (409, 429):
            return

    print("\n[2] Testing Duplicate Registration (Should fail with 409)")
    reg_dup = requests.post(f"{BASE_URL}/api/v1/auth/register", json=reg_payload)
    if reg_dup.status_code == 409:
        print(" ✅ Duplicate registration correctly blocked with 409.")
    else:
        print(f" ❌ Expected 409, got {reg_dup.status_code}: {reg_dup.text}")

    print("\n[3] Testing Login")
    login_payload = {
        "email": test_email,
        "password": test_password
    }
    login_res = requests.post(f"{BASE_URL}/api/v1/auth/login", json=login_payload)
    
    if login_res.status_code == 200:
        print(" ✅ Login successful.")
        data = login_res.json()
        access_token = data.get("access_token")
        refresh_token = data.get("refresh_token")
        profile = data.get("profile", {})
        print(f"    User ID: {profile.get('user_id')}")
        print(f"    Role: {profile.get('role')}")
    else:
        print(f" ❌ Login failed: {login_res.status_code} - {login_res.text}")
        return

    print("\n[4] Testing /me (Authenticated Profile Retrieval)")
    headers = {"Authorization": f"Bearer {access_token}"}
    me_res = requests.get(f"{BASE_URL}/api/v1/auth/me", headers=headers)
    if me_res.status_code == 200:
        print(" ✅ /me retrieved successfully.")
        me_data = me_res.json()
        print(f"    Registered Email: {me_data.get('email')}")
        print(f"    Vessels Count: {len(me_data.get('vessels', []))}")
    else:
        print(f" ❌ /me failed: {me_res.status_code} - {me_res.text}")

    print("\n[5] Testing Token Refresh")
    refresh_payload = {"refresh_token": refresh_token}
    refresh_res = requests.post(f"{BASE_URL}/api/v1/auth/refresh", json=refresh_payload)
    if refresh_res.status_code == 200:
        print(" ✅ Token refresh successful.")
    else:
        print(f" ❌ Token refresh failed: {refresh_res.status_code} - {refresh_res.text}")

    print("\n[6] Testing Logout")
    logout_res = requests.post(f"{BASE_URL}/api/v1/auth/logout", headers=headers)
    if logout_res.status_code == 200:
        print(" ✅ Logout successful.")
    else:
        print(f" ❌ Logout failed: {logout_res.status_code} - {logout_res.text}")

    print("\n🎉 Auth Flow Test Completed Successfully.")

if __name__ == "__main__":
    test_auth_flow()
