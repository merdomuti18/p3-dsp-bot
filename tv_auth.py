"""
TradingView oturum yardımcısı (tvDatafeed).

tvDatafeed varsayılan login'i User-Agent göndermez; TradingView sıkça
reddedip `error while signin` basar, client ise anonymous token ile devam eder.

Öncelik:
  1) TV_AUTH_TOKEN
  2) TV_SESSIONID (+ isteğe bağlı TV_SESSIONID_SIGN) — Application → Cookies
  3) TV_USERNAME + TV_PASSWORD (2FA açıksa çalışmaz)

En kolay yol (2FA ile):
  DevTools → Uygulama → Çerezler → https://www.tradingview.com
  → `sessionid` ve varsa `sessionid_sign` değerlerini kopyala
  → sohbete yapıştırma; secret set et:
       gh secret set TV_SESSIONID -R merdomuti18/p3-dsp-bot
       gh secret set TV_SESSIONID_SIGN -R merdomuti18/p3-dsp-bot
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

import requests

logger = logging.getLogger(__name__)

SIGNIN_URL = "https://www.tradingview.com/accounts/signin/"
UNAUTHORIZED = "unauthorized_user_token"
_AUTH_TOKEN_RE = re.compile(r'"auth_token"\s*:\s*"([^"]+)"')
_SIGNIN_HEADERS = {
    "Referer": "https://www.tradingview.com",
    "Origin": "https://www.tradingview.com",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/128.0.0.0 Safari/537.36"
    ),
}


def _env(*names: str) -> str:
    for name in names:
        val = (os.environ.get(name) or "").strip()
        if val:
            return val
    return ""


def signin_password(username: str, password: str) -> tuple[str | None, dict[str, Any]]:
    """POST /accounts/signin/. Returns (auth_token|None, safe diagnostics)."""
    diag: dict[str, Any] = {
        "ok": False,
        "http_status": None,
        "error": None,
        "code": None,
        "has_user": False,
        "user_len": len(username),
        "pass_len": len(password),
    }
    try:
        resp = requests.post(
            SIGNIN_URL,
            data={"username": username, "password": password, "remember": "on"},
            headers=_SIGNIN_HEADERS,
            timeout=60,
        )
        diag["http_status"] = resp.status_code
        try:
            payload = resp.json()
        except Exception:
            diag["error"] = f"non_json_body status={resp.status_code}"
            return None, diag

        if not isinstance(payload, dict):
            diag["error"] = "unexpected_payload_type"
            return None, diag

        diag["error"] = payload.get("error") or payload.get("code")
        diag["code"] = payload.get("code")
        user = payload.get("user")
        diag["has_user"] = isinstance(user, dict)
        if isinstance(user, dict):
            token = user.get("auth_token")
            if token:
                diag["ok"] = True
                diag["error"] = None
                return str(token), diag
        return None, diag
    except Exception as exc:
        diag["error"] = f"request_failed:{type(exc).__name__}"
        return None, diag


def token_from_sessionid(
    sessionid: str,
    sessionid_sign: str = "",
) -> tuple[str | None, dict[str, Any]]:
    """Use browser sessionid cookie(s) to pull auth_token from TradingView HTML."""
    diag: dict[str, Any] = {
        "ok": False,
        "source": "sessionid",
        "http_status": None,
        "error": None,
        "code": None,
        "has_user": False,
        "user_len": 0,
        "pass_len": 0,
        "sessionid_len": len(sessionid),
        "sessionid_sign_len": len(sessionid_sign),
    }
    try:
        sess = requests.Session()
        sess.cookies.set("sessionid", sessionid, domain=".tradingview.com", path="/")
        if sessionid_sign:
            sess.cookies.set(
                "sessionid_sign",
                sessionid_sign,
                domain=".tradingview.com",
                path="/",
            )
        resp = sess.get(
            "https://www.tradingview.com/",
            headers=_SIGNIN_HEADERS,
            timeout=60,
        )
        diag["http_status"] = resp.status_code
        match = _AUTH_TOKEN_RE.search(resp.text or "")
        if not match:
            diag["error"] = "auth_token_not_in_page"
            return None, diag
        diag["ok"] = True
        return match.group(1), diag
    except Exception as exc:
        diag["error"] = f"request_failed:{type(exc).__name__}"
        return None, diag


def resolve_auth_token() -> tuple[str | None, dict[str, Any]]:
    """Resolve token: TV_AUTH_TOKEN → sessionid → username/password."""
    direct = _env("TV_AUTH_TOKEN", "TRADINGVIEW_AUTH_TOKEN")
    if direct:
        return direct, {
            "ok": True,
            "source": "TV_AUTH_TOKEN",
            "user_len": 0,
            "pass_len": 0,
            "error": None,
            "http_status": None,
            "code": None,
            "has_user": False,
        }

    sessionid = _env("TV_SESSIONID", "TRADINGVIEW_SESSIONID")
    if sessionid:
        sign = _env("TV_SESSIONID_SIGN", "TRADINGVIEW_SESSIONID_SIGN")
        token, diag = token_from_sessionid(sessionid, sign)
        return token, diag

    user = _env("TV_USERNAME", "TRADINGVIEW_USERNAME")
    pw = _env("TV_PASSWORD", "TRADINGVIEW_PASSWORD")
    if not user or not pw:
        return None, {
            "ok": False,
            "source": "missing",
            "error": "Set TV_SESSIONID (easiest) or TV_AUTH_TOKEN or TV_USERNAME/PASSWORD",
            "user_len": len(user),
            "pass_len": len(pw),
            "http_status": None,
            "code": None,
            "has_user": False,
        }

    token, diag = signin_password(user, pw)
    diag["source"] = "password"
    return token, diag


def make_tv_client(*, require_auth: bool = True):
    """
    Build an authenticated TvDatafeed client.

    Sets auth_token explicitly (bypasses broken stock __auth UA).
    """
    from tvDatafeed import TvDatafeed

    token, diag = resolve_auth_token()
    if not token:
        msg = f"TV auth failed: {diag}"
        if require_auth:
            raise RuntimeError(msg)
        logger.warning(msg)
        return TvDatafeed(), diag

    client = TvDatafeed()  # starts anonymous
    client.token = token
    if getattr(client, "token", None) in (None, UNAUTHORIZED):
        raise RuntimeError(f"TV token inject failed: {diag}")
    diag["token_set"] = True
    logger.info(
        "TV auth OK source=%s user_len=%s",
        diag.get("source"),
        diag.get("user_len"),
    )
    return client, diag


def is_authenticated(client) -> bool:
    tok = getattr(client, "token", None)
    return bool(tok) and tok != UNAUTHORIZED
