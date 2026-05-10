"""Manage Chainlit password users for local ChainAgents deployments."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CHAINLIT_AUTH_USERS_FILE_ENV = "CHAINLIT_AUTH_USERS_FILE"
DEFAULT_USERS_FILE = Path(".files/users.json")
HASH_ALGORITHM = "pbkdf2_sha256"
HASH_ITERATIONS = 600_000
STORE_VERSION = 1
USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_.@-]{1,128}$")


class UserStoreError(ValueError):
    """Raised when the Chainlit users file cannot be used safely."""


@dataclass(frozen=True)
class ChainlitUserRecord:
    """Represent one configured Chainlit password user."""

    username: str
    display_name: str


def resolve_users_file(path: str | os.PathLike[str] | None = None) -> Path:
    """Resolve a Chainlit auth users file path.

    Args:
        path: Explicit path value.

    Returns:
        The resolved user file path.
    """
    raw_path = str(path or os.getenv(CHAINLIT_AUTH_USERS_FILE_ENV) or DEFAULT_USERS_FILE)
    return Path(raw_path).expanduser().resolve()


def normalize_username(username: str) -> str:
    """Normalize and validate a user name.

    Args:
        username: Raw username.

    Returns:
        The normalized username.

    Raises:
        UserStoreError: If the username is unsupported.
    """
    candidate = username.strip()
    if not candidate:
        raise UserStoreError("Username cannot be empty.")
    if not USERNAME_PATTERN.fullmatch(candidate):
        raise UserStoreError(
            "Username must use only letters, numbers, dot, underscore, at sign, or dash."
        )
    return candidate


def validate_password(password: str) -> str:
    """Validate a password supplied for a local Chainlit user.

    Args:
        password: Raw password.

    Returns:
        The password.

    Raises:
        UserStoreError: If the password is unsupported.
    """
    if len(password) < 8:
        raise UserStoreError("Password must be at least 8 characters.")
    return password


def hash_password(password: str) -> str:
    """Return a salted password hash.

    Args:
        password: Plain-text password.

    Returns:
        Encoded password hash.
    """
    validate_password(password)
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        HASH_ITERATIONS,
    )
    encoded_salt = base64.urlsafe_b64encode(salt).decode("ascii")
    encoded_digest = base64.urlsafe_b64encode(digest).decode("ascii")
    return f"{HASH_ALGORITHM}${HASH_ITERATIONS}${encoded_salt}${encoded_digest}"


def verify_password(password: str, password_hash: str) -> bool:
    """Return whether a plain-text password matches an encoded hash.

    Args:
        password: Plain-text password.
        password_hash: Encoded password hash.

    Returns:
        Whether the password matches.
    """
    parts = password_hash.split("$")
    if len(parts) != 4:
        return False
    algorithm, raw_iterations, encoded_salt, encoded_digest = parts
    if algorithm != HASH_ALGORITHM:
        return False
    try:
        iterations = int(raw_iterations)
        salt = base64.urlsafe_b64decode(encoded_salt.encode("ascii"))
        expected = base64.urlsafe_b64decode(encoded_digest.encode("ascii"))
    except (TypeError, ValueError):
        return False
    actual = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        iterations,
    )
    return secrets.compare_digest(actual, expected)


def load_user_store(path: str | os.PathLike[str] | Path) -> dict[str, Any]:
    """Load a Chainlit user store.

    Args:
        path: User store path.

    Returns:
        The loaded store payload.

    Raises:
        UserStoreError: If the store cannot be parsed.
    """
    users_file = Path(path)
    if not users_file.exists():
        return {"version": STORE_VERSION, "users": {}}
    try:
        payload = json.loads(users_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise UserStoreError(f"Invalid users file JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise UserStoreError("Users file must contain a JSON object.")
    users = payload.get("users")
    if not isinstance(users, dict):
        raise UserStoreError("Users file must contain a 'users' object.")
    return {"version": payload.get("version", STORE_VERSION), "users": users}


def save_user_store(path: str | os.PathLike[str] | Path, payload: dict[str, Any]) -> None:
    """Write a Chainlit user store atomically.

    Args:
        path: User store path.
        payload: Store payload.
    """
    users_file = Path(path)
    users_file.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, indent=2, sort_keys=True)
    tmp_path = users_file.with_name(f".{users_file.name}.{os.getpid()}.tmp")
    tmp_path.write_text(f"{body}\n", encoding="utf-8")
    os.chmod(tmp_path, 0o600)
    os.replace(tmp_path, users_file)
    os.chmod(users_file, 0o600)


def _record_from_store(username: str, value: Any) -> ChainlitUserRecord | None:
    if not isinstance(value, dict):
        return None
    display_name = str(value.get("display_name") or username).strip() or username
    return ChainlitUserRecord(username=username, display_name=display_name)


def list_users(path: str | os.PathLike[str] | Path) -> list[ChainlitUserRecord]:
    """List configured Chainlit users.

    Args:
        path: User store path.

    Returns:
        User records sorted by username.
    """
    payload = load_user_store(path)
    users = payload["users"]
    records = [
        record
        for username, value in sorted(users.items())
        if (record := _record_from_store(str(username), value)) is not None
    ]
    return records


def add_user(
    path: str | os.PathLike[str] | Path,
    *,
    username: str,
    password: str,
    display_name: str | None = None,
    overwrite: bool = False,
) -> ChainlitUserRecord:
    """Add or replace a Chainlit password user.

    Args:
        path: User store path.
        username: Username to add.
        password: Plain-text password to hash.
        display_name: Optional display name.
        overwrite: Whether to replace an existing user.

    Returns:
        The added user record.

    Raises:
        UserStoreError: If the user already exists or data is invalid.
    """
    normalized_username = normalize_username(username)
    normalized_display_name = (display_name or normalized_username).strip()
    if not normalized_display_name:
        normalized_display_name = normalized_username
    payload = load_user_store(path)
    users = payload["users"]
    if normalized_username in users and not overwrite:
        raise UserStoreError(f"User '{normalized_username}' already exists.")
    record = {
        "display_name": normalized_display_name,
        "password_hash": hash_password(password),
    }
    users[normalized_username] = record
    payload["version"] = STORE_VERSION
    save_user_store(path, payload)
    return ChainlitUserRecord(
        username=normalized_username,
        display_name=normalized_display_name,
    )


def remove_user(path: str | os.PathLike[str] | Path, username: str) -> bool:
    """Remove a Chainlit password user.

    Args:
        path: User store path.
        username: Username to remove.

    Returns:
        Whether a user was removed.
    """
    normalized_username = normalize_username(username)
    payload = load_user_store(path)
    users = payload["users"]
    if normalized_username not in users:
        return False
    del users[normalized_username]
    payload["version"] = STORE_VERSION
    save_user_store(path, payload)
    return True


def authenticate_user(
    path: str | os.PathLike[str] | Path,
    username: str,
    password: str,
) -> ChainlitUserRecord | None:
    """Authenticate a user against a Chainlit users file.

    Args:
        path: User store path.
        username: Username to authenticate.
        password: Plain-text password.

    Returns:
        The authenticated record, if credentials are valid.
    """
    try:
        normalized_username = normalize_username(username)
        payload = load_user_store(path)
    except UserStoreError:
        return None
    value = payload["users"].get(normalized_username)
    if not isinstance(value, dict):
        return None
    password_hash = str(value.get("password_hash") or "")
    if not password_hash or not verify_password(password, password_hash):
        return None
    return _record_from_store(normalized_username, value)
