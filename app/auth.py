# app/auth.py
"""
Cognito authentication with client-secret + NEW_PASSWORD_REQUIRED support.
"""
from __future__ import annotations

import base64
import hashlib
import hmac

import boto3
import streamlit as st
from botocore.exceptions import ClientError


def _region() -> str:
    try:
        return st.secrets["aws"]["region_name"]
    except Exception:
        return "us-east-1"


def _secret_hash(username: str, client_id: str, client_secret: str) -> str:
    """Compute Cognito SECRET_HASH for USER_PASSWORD_AUTH and challenges."""
    msg = (username + client_id).encode("utf-8")
    key = client_secret.encode("utf-8")
    digest = hmac.new(key, msg, hashlib.sha256).digest()
    return base64.b64encode(digest).decode("utf-8")


def get_cognito_client():
    """Create cognito-idp client. For prod, prefer role-based creds; secrets are fine for local dev."""
    try:
        aws = st.secrets["aws"]
        return boto3.client(
            "cognito-idp",
            aws_access_key_id=aws.get("iam_user_access_key_id"),
            aws_secret_access_key=aws.get("iam_user_secret_access_key"),
            region_name=_region(),
        )
    except Exception:
        st.error("Missing AWS config. Ensure `[aws]` in `.streamlit/secrets.toml`.")
        return None


def _clear_auth_state():
    for k in [
        "logged_in",
        "username",
        "cognito_tokens",
        "auth_challenge",
        "auth_session",
        "required_attributes",
    ]:
        st.session_state.pop(k, None)


def logout():
    _clear_auth_state()
    st.rerun()


def start_login(username: str, password: str) -> bool:
    """
    Step 1: initiate auth. If NEW_PASSWORD_REQUIRED, store challenge state and return False
    so the UI can render the new-password form.
    """
    client = get_cognito_client()
    if not client:
        return False

    try:
        app_client_id = st.secrets["aws"]["cognito_app_client_id"]
        client_secret = st.secrets["aws"].get("cognito_app_client_secret")

        auth_params = {"USERNAME": username, "PASSWORD": password}
        if client_secret:
            auth_params["SECRET_HASH"] = _secret_hash(username, app_client_id, client_secret)

        resp = client.initiate_auth(
            ClientId=app_client_id,
            AuthFlow="USER_PASSWORD_AUTH",
            AuthParameters=auth_params,
        )

        # Normal success (no challenge)
        if "AuthenticationResult" in resp:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.cognito_tokens = resp["AuthenticationResult"]
            st.session_state.pop("auth_challenge", None)
            st.session_state.pop("auth_session", None)
            st.session_state.pop("required_attributes", None)
            st.rerun()
            return True

        # Challenge flow (first login)
        if resp.get("ChallengeName") == "NEW_PASSWORD_REQUIRED":
            st.session_state.auth_challenge = "NEW_PASSWORD_REQUIRED"
            st.session_state.auth_session = resp.get("Session")
            st.session_state.username = username
            st.session_state.required_attributes = resp.get("ChallengeParameters", {}).get(
                "requiredAttributes", "[]"
            )
            # Do NOT rerun; let UI show the "set new password" form
            return False

        st.error(f"Unsupported auth challenge: {resp.get('ChallengeName')}")
        return False

    except ClientError as e:
        code = e.response["Error"]["Code"]
        msg = e.response["Error"].get("Message", code)
        # TEMP: show the real cause while you’re wiring things up
        st.error(f"{code}: {msg}")
        return False
    except Exception as e:
        st.error(f"Unexpected login error: {str(e)}")
        return False


def complete_new_password(new_password: str, user_attributes: dict | None = None) -> bool:
    """
    Step 2: complete NEW_PASSWORD_REQUIRED. Provide the new password (and any required attrs if needed).
    """
    client = get_cognito_client()
    if not client:
        return False

    session = st.session_state.get("auth_session")
    username = st.session_state.get("username")
    if not session or not username:
        st.error("Challenge session expired. Please login again.")
        _clear_auth_state()
        return False

    try:
        app_client_id = st.secrets["aws"]["cognito_app_client_id"]
        client_secret = st.secrets["aws"].get("cognito_app_client_secret")

        challenge_responses = {
            "USERNAME": username,
            "NEW_PASSWORD": new_password,
        }
        if client_secret:
            challenge_responses["SECRET_HASH"] = _secret_hash(username, app_client_id, client_secret)

        # If the pool enforces required attributes at first login, add them:
        if user_attributes:
            for k, v in user_attributes.items():
                challenge_responses[k] = v

        resp = client.respond_to_auth_challenge(
            ClientId=app_client_id,
            ChallengeName="NEW_PASSWORD_REQUIRED",
            Session=session,
            ChallengeResponses=challenge_responses,
        )

        if "AuthenticationResult" in resp:
            st.session_state.logged_in = True
            st.session_state.cognito_tokens = resp["AuthenticationResult"]
            st.session_state.pop("auth_challenge", None)
            st.session_state.pop("auth_session", None)
            st.session_state.pop("required_attributes", None)
            st.success("Password updated. You are now logged in.")
            st.rerun()
            return True

        st.error("Unexpected response while completing password change.")
        return False

    except ClientError as e:
        code = e.response["Error"]["Code"]
        msg = e.response["Error"].get("Message", code)
        st.error(f"{code}: {msg}")
        return False
    except Exception as e:
        st.error(f"Unexpected error while setting new password: {str(e)}")
        return False