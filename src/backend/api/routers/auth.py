from __future__ import annotations

import logging
import random
import string
import time

from api.dependencies import require_user
from core.config import demo_auth_bypass_enabled
from core.firebase import is_firebase_enabled
from core.rate_limit import AUTH_TOKEN_LIMIT, limiter
from core.security import hash_password, hash_token, is_valid_email
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

logger = logging.getLogger("vrptw.auth")

router = APIRouter(tags=["auth"])


# ── Pydantic request schemas ──────────────────────────────

class OtpRequestBody(BaseModel):
    email: str

class OtpVerifyBody(BaseModel):
    email: str
    otp: str

class RegisterBody(BaseModel):
    email: str
    password: str
    otp: str

class ForgotPasswordRequestBody(BaseModel):
    email: str

class ForgotPasswordResetBody(BaseModel):
    token: str
    password: str


# ── Helper: guard endpoints requiring Firestore ──────────

def _require_firestore():
    """Raise 503 if Firestore-backed auth is not available."""
    if not is_firebase_enabled() and not demo_auth_bypass_enabled():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Auth service unavailable — Firebase is not configured.",
        )


def _generate_otp(length: int = 6) -> str:
    """Generate a numeric OTP code."""
    return "".join(random.choices(string.digits, k=length))


# ── Existing endpoints ────────────────────────────────────

@router.get("/auth/me")
async def auth_me(user: dict[str, str] = Depends(require_user)) -> dict[str, str]:
    """Returns the verified user context from the Firebase token."""
    return user


@router.post("/auth/token")
@limiter.limit(AUTH_TOKEN_LIMIT)
async def auth_token(request: Request) -> dict[str, str]:
    """Stub endpoint for authentication token requests, primarily for compatibility and rate limiting tests."""
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Password authentication is disabled. Please use Firebase ID token auth.",
    )


# ── Registration OTP Flow ─────────────────────────────────

@router.post("/auth/register/request-otp")
async def register_request_otp(body: OtpRequestBody):
    """Generate and store a 6-digit OTP for email verification during registration."""
    _require_firestore()
    email = body.email.strip().lower()
    if not is_valid_email(email):
        raise HTTPException(status_code=422, detail="Invalid email format.")

    try:
        from database.repositories.otp_repo import upsert_register_otp

        otp_code = _generate_otp()
        otp_hashed = hash_token(otp_code)
        now_ts = int(time.time())
        expires_at = now_ts + 600  # 10 minutes

        upsert_register_otp(email, otp_hashed, expires_at, now_ts)
        logger.info("OTP generated for %s: %s (dev only — remove in production)", email, otp_code)
        return {"ok": True, "message": f"OTP sent to {email}.", "otp_dev": otp_code}
    except Exception as exc:
        logger.warning("register_request_otp failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/auth/register/verify-otp")
async def register_verify_otp(body: OtpVerifyBody):
    """Verify the OTP code submitted by the user."""
    _require_firestore()
    email = body.email.strip().lower()
    otp = body.otp.strip()

    try:
        from database.repositories.otp_repo import find_register_otp

        record = find_register_otp(email)
        if not record:
            raise HTTPException(status_code=400, detail="No OTP request found for this email.")

        now_ts = int(time.time())
        if int(record.get("expires_at", 0)) < now_ts:
            raise HTTPException(status_code=400, detail="OTP has expired. Please request a new one.")

        if hash_token(otp) != record.get("otp_hash", ""):
            raise HTTPException(status_code=400, detail="Incorrect OTP code.")

        return {"ok": True, "message": "OTP verified."}
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("register_verify_otp failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/auth/register")
async def register(body: RegisterBody):
    """Create a new user account after OTP verification."""
    _require_firestore()
    email = body.email.strip().lower()
    password = body.password.strip()
    otp = body.otp.strip()

    if not is_valid_email(email):
        raise HTTPException(status_code=422, detail="Invalid email format.")
    if len(password) < 6:
        raise HTTPException(status_code=422, detail="Password must be at least 6 characters.")

    try:
        from database.repositories.otp_repo import delete_register_otp, find_register_otp
        from database.repositories.users_repo import create_user, find_user_by_email

        # Verify OTP one more time
        record = find_register_otp(email)
        if not record:
            raise HTTPException(status_code=400, detail="No OTP verified for this email.")
        if hash_token(otp) != record.get("otp_hash", ""):
            raise HTTPException(status_code=400, detail="Invalid OTP.")
        if int(record.get("expires_at", 0)) < int(time.time()):
            raise HTTPException(status_code=400, detail="OTP expired.")

        # Check for existing user
        existing = find_user_by_email(email)
        if existing:
            raise HTTPException(status_code=409, detail="An account with this email already exists.")

        # Create user
        pw_hash = hash_password(password)
        now_ts = int(time.time())
        create_user(email, pw_hash, "operator", now_ts, must_change_password=False)
        delete_register_otp(email)

        return {"ok": True, "message": "Account created successfully."}
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("register failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ── Forgot Password Flow ──────────────────────────────────

@router.post("/auth/forgot-password/request")
async def forgot_password_request(body: ForgotPasswordRequestBody):
    """Generate a password reset token for the given email."""
    _require_firestore()
    email = body.email.strip().lower()
    if not is_valid_email(email):
        raise HTTPException(status_code=422, detail="Invalid email format.")

    try:
        from database.repositories.otp_repo import replace_password_reset_token
        from database.repositories.users_repo import find_user_by_email

        user = find_user_by_email(email)
        if not user:
            # Don't reveal whether the email exists — return success either way
            return {"ok": True, "message": "If an account exists, a reset link has been sent."}

        token_raw = "".join(random.choices(string.ascii_letters + string.digits, k=48))
        token_hashed = hash_token(token_raw)
        now_ts = int(time.time())
        expires_at = now_ts + 3600  # 1 hour

        replace_password_reset_token(email, token_hashed, expires_at, now_ts)
        logger.info("Password reset token for %s: %s (dev only)", email, token_raw)

        return {"ok": True, "message": "If an account exists, a reset link has been sent.", "token_dev": token_raw}
    except Exception as exc:
        logger.warning("forgot_password_request failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/auth/forgot-password/reset")
async def forgot_password_reset(body: ForgotPasswordResetBody):
    """Reset the user's password using a valid reset token."""
    _require_firestore()
    password = body.password.strip()
    if len(password) < 6:
        raise HTTPException(status_code=422, detail="Password must be at least 6 characters.")

    try:
        from database.repositories.otp_repo import (
            find_valid_password_reset_token,
            mark_password_reset_token_used,
        )
        from database.repositories.users_repo import update_user_password

        token_hashed = hash_token(body.token.strip())
        now_ts = int(time.time())
        record = find_valid_password_reset_token(token_hashed, now_ts)
        if not record:
            raise HTTPException(status_code=400, detail="Invalid or expired reset token.")

        email = record.get("email", "")
        pw_hash = hash_password(password)
        update_user_password(email, pw_hash, must_change_password=False)
        mark_password_reset_token_used(token_hashed)

        return {"ok": True, "message": "Password has been reset successfully."}
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("forgot_password_reset failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
