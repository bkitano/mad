"""Supabase JWT authentication for FastAPI."""

import os
import httpx
from fastapi import Depends, HTTPException, Request


SUPABASE_URL = os.environ.get("SUPABASE_URL", "")


def _get_token(request: Request) -> str:
    """Extract Bearer token from the Authorization header."""
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    return auth[7:]


async def get_current_user(token: str = Depends(_get_token)) -> dict:
    """Validate a Supabase access token and return the user.

    Calls Supabase's GoTrue /auth/v1/user endpoint which validates the JWT
    and returns the user object.  Returns a dict with at least 'id' and 'email'.
    """
    if not SUPABASE_URL:
        raise HTTPException(status_code=500, detail="SUPABASE_URL not configured")

    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{SUPABASE_URL}/auth/v1/user",
            headers={
                "Authorization": f"Bearer {token}",
                "apikey": os.environ.get("SUPABASE_KEY", ""),
            },
        )

    if resp.status_code != 200:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user = resp.json()
    if not user.get("id"):
        raise HTTPException(status_code=401, detail="Could not resolve user from token")

    return {"id": user["id"], "email": user.get("email", "")}
