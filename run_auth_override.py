"""
Temporary auth tester that reads BASE_URL from environment (default http://127.0.0.1:8001).
Used to run the same auth flow against a custom port when 8000 is occupied.
"""

import os
import uuid
import time
import requests

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8001")


def test_auth_flow_once():
    unique_id = str(uuid.uuid4())[:8]
    test_email = f"testuser_{unique_id}@example.com"
    test_password = "SecurePassword123!"

    reg_payload = {
        "email": test_email,
        "password": test_password,
        "full_name": "Test Fisherman",
        "role": "fisherman",
        "phone": f"+9198765{int(time.time()) % 10000:04d}",
        "preferred_language": "en",
    }

    try:
        reg_res = requests.post(f"{BASE_URL}/api/v1/auth/register", json=reg_payload, timeout=10)
    except requests.exceptions.ConnectionError:
        print("CONNERR")
        return False, "connection"
    except Exception as e:
        print("ERR", e)
        return False, str(e)

    if reg_res.status_code not in (201, 409, 429):
        print(f"REG_FAIL {reg_res.status_code} {reg_res.text}")
        return False, f"reg:{reg_res.status_code}"

    login_payload = {"email": test_email, "password": test_password}
    login_res = requests.post(f"{BASE_URL}/api/v1/auth/login", json=login_payload, timeout=10)
    if login_res.status_code != 200:
        print(f"LOGIN_FAIL {login_res.status_code} {login_res.text}")
        return False, f"login:{login_res.status_code}"

    data = login_res.json()
    access_token = data.get("access_token")
    if not access_token:
        print("NO_TOKEN")
        return False, "no_token"

    headers = {"Authorization": f"Bearer {access_token}"}
    me_res = requests.get(f"{BASE_URL}/api/v1/auth/me", headers=headers, timeout=10)
    if me_res.status_code != 200:
        print(f"ME_FAIL {me_res.status_code} {me_res.text}")
        return False, f"me:{me_res.status_code}"

    return True, "ok"


if __name__ == "__main__":
    ok, reason = test_auth_flow_once()
    print(ok, reason)
