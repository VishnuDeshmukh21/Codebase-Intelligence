# app/auth.py
"""
Authentication, user management, session tracking, rate limiting,
and usage analytics for the Codebase Intelligence Agent.
"""
from __future__ import annotations

import hashlib
import os
import time
from datetime import datetime, timezone
from typing import Any, Optional

import structlog
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pymongo.database import Database

log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Security Configuration — read from environment, never hardcoded
# ---------------------------------------------------------------------------
SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "CHANGE_ME_IN_PROD_USE_ENV_VAR_32chars!!")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("JWT_EXPIRE_MINUTES", "1440"))  # 24h

# Rate limiting: normal users
RATE_LIMIT_MAX_MESSAGES = 4          # max messages per window (shown to user)
RATE_LIMIT_WINDOW_SECONDS = 60       # rolling 60-second window

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer_scheme = HTTPBearer(auto_error=False)


# ---------------------------------------------------------------------------
# Password Utilities
# ---------------------------------------------------------------------------

def hash_password(plain: str) -> str:
    return pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


# ---------------------------------------------------------------------------
# JWT Utilities
# ---------------------------------------------------------------------------

def create_access_token(payload: dict) -> str:
    import copy
    data = copy.deepcopy(payload)
    data["iat"] = int(time.time())
    data["exp"] = int(time.time()) + ACCESS_TOKEN_EXPIRE_MINUTES * 60
    return jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError as exc:
        raise HTTPException(status_code=401, detail=f"Invalid or expired token: {exc}")


# ---------------------------------------------------------------------------
# User CRUD (MongoDB collection: users)
# ---------------------------------------------------------------------------

def get_users_col(db: Database):
    col = db["users"]
    col.create_index("email", unique=True)
    return col


def get_user_by_email(db: Database, email: str) -> Optional[dict]:
    return get_users_col(db).find_one({"email": email.lower()})


def create_user(db: Database, email: str, password: str, name: str) -> dict:
    col = get_users_col(db)
    if col.find_one({"email": email.lower()}):
        raise HTTPException(status_code=409, detail="Email already registered.")
    doc = {
        "email": email.lower(),
        "name": name.strip(),
        "password_hash": hash_password(password),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_messages": 0,
        "total_tokens": 0,
        "total_cost_usd": 0.0,
    }
    col.insert_one(doc)
    return {k: v for k, v in doc.items() if k != "password_hash"}


# ---------------------------------------------------------------------------
# VIP / Special Users — fetched from DB at runtime, never hardcoded
# ---------------------------------------------------------------------------

def get_vip_users(db: Database) -> list[str]:
    """
    Returns a list of VIP user names. Stored in MongoDB collection `app_config`
    so they can be managed without a code deploy.
    """
    doc = db["app_config"].find_one({"_id": "vip_users"})
    if doc:
        return doc.get("names", [])
    return []


def seed_vip_users(db: Database, names: list[str]):
    """Seed VIP users into the config collection (idempotent)."""
    db["app_config"].update_one(
        {"_id": "vip_users"},
        {"$set": {"names": names}},
        upsert=True,
    )


def is_vip_user(db: Database, user_name: str) -> bool:
    vips = get_vip_users(db)
    name_lower = user_name.strip().lower()
    return any(v.strip().lower() == name_lower for v in vips)


# ---------------------------------------------------------------------------
# Rate Limiting (in-memory sliding window per user_id)
# ---------------------------------------------------------------------------

_rate_limit_store: dict[str, list[float]] = {}


def check_rate_limit(user_id: str, is_vip: bool = False) -> dict:
    """
    Returns {"allowed": bool, "remaining": int, "reset_in": float, "limit": int}.
    VIP users bypass rate limiting entirely.
    """
    if is_vip:
        return {"allowed": True, "remaining": 999, "reset_in": 0, "limit": 999}

    now = time.time()
    window = _rate_limit_store.get(user_id, [])
    # Prune old timestamps outside the sliding window
    window = [t for t in window if now - t < RATE_LIMIT_WINDOW_SECONDS]

    remaining = RATE_LIMIT_MAX_MESSAGES - len(window)
    reset_in = (RATE_LIMIT_WINDOW_SECONDS - (now - window[0])) if window else 0

    if len(window) >= RATE_LIMIT_MAX_MESSAGES:
        _rate_limit_store[user_id] = window
        return {
            "allowed": False,
            "remaining": 0,
            "reset_in": round(reset_in, 1),
            "limit": RATE_LIMIT_MAX_MESSAGES,
        }

    window.append(now)
    _rate_limit_store[user_id] = window
    return {
        "allowed": True,
        "remaining": remaining - 1,
        "reset_in": round(reset_in, 1),
        "limit": RATE_LIMIT_MAX_MESSAGES,
    }


# ---------------------------------------------------------------------------
# Session & Usage Tracking (MongoDB collection: sessions)
# ---------------------------------------------------------------------------

def get_sessions_col(db: Database):
    col = db["sessions"]
    col.create_index("user_id")
    col.create_index("thread_id", unique=True, sparse=True)
    return col


def create_session(db: Database, user_id: str, user_name: str, repo_id: str, thread_id: str) -> str:
    col = get_sessions_col(db)
    doc = {
        "thread_id": thread_id,
        "user_id": user_id,
        "user_name": user_name,
        "repo_id": repo_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "last_active": datetime.now(timezone.utc).isoformat(),
        "messages": 0,
        "tokens_used": 0,
        "cost_usd": 0.0,
        "title": "",          # set on first message
        "status": "active",
    }
    col.update_one({"thread_id": thread_id}, {"$set": doc}, upsert=True)
    return thread_id


def update_session_usage(
    db: Database,
    thread_id: str,
    tokens: int = 0,
    cost_usd: float = 0.0,
    title: str = "",
):
    col = get_sessions_col(db)
    update = {
        "$inc": {"messages": 1, "tokens_used": tokens, "cost_usd": cost_usd},
        "$set": {"last_active": datetime.now(timezone.utc).isoformat(), "status": "active"},
    }
    if title:
        update["$setOnInsert"] = {"title": title}
        col.update_one({"thread_id": thread_id}, {"$setOnInsert": {"title": title}}, upsert=False)
    col.update_one({"thread_id": thread_id}, {"$inc": {"messages": 1, "tokens_used": tokens, "cost_usd": cost_usd}, "$set": {"last_active": datetime.now(timezone.utc).isoformat()}})


def get_user_sessions(db: Database, user_id: str) -> list[dict]:
    col = get_sessions_col(db)
    return list(col.find({"user_id": user_id}, {"_id": 0}).sort("last_active", -1))


def log_user_message(db: Database, thread_id: str, user_id: str, task: str):
    """Record each message in session and increment user counters."""
    col = get_sessions_col(db)
    col.update_one(
        {"thread_id": thread_id},
        {
            "$inc": {"messages": 1},
            "$set": {
                "last_active": datetime.now(timezone.utc).isoformat(),
                "title": task[:60] if task else "",
            },
        },
        upsert=False,
    )


# ---------------------------------------------------------------------------
# FastAPI Dependency: Authenticated User
# ---------------------------------------------------------------------------

def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> dict:
    """Dependency that extracts and validates the current authenticated user."""
    token = None

    # Try Authorization header first
    if credentials and credentials.credentials:
        token = credentials.credentials

    # Fallback to query param (for SSE streams that can't set headers easily)
    if not token:
        token = request.query_params.get("token")

    if not token:
        raise HTTPException(status_code=401, detail="Authentication required. Please log in.")

    payload = decode_access_token(token)
    return payload

