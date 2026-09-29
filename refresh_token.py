"""Refreshes a long-lived Instagram Login token (valid 60 days) and prints the new one."""
import os
import requests

r = requests.get(
    "https://graph.instagram.com/refresh_access_token",
    params={"grant_type": "ig_refresh_token", "access_token": os.environ["IG_ACCESS_TOKEN"]},
    timeout=30,
)
r.raise_for_status()
print(r.json()["access_token"])
