from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

import chainagents_cli
import main
from chainlit_users import (
    UserStoreError,
    add_user,
    authenticate_user,
    list_users,
    remove_user,
    resolve_users_file,
)


def test_user_store_add_authenticate_list_and_remove_user(tmp_path: Path) -> None:
    users_file = tmp_path / "users.json"

    added = add_user(
        users_file,
        username="alice",
        password="correct horse battery staple",
        display_name="Alice Example",
    )

    assert added.username == "alice"
    assert added.display_name == "Alice Example"
    assert list_users(users_file) == [added]

    raw_store = json.loads(users_file.read_text())
    assert "correct horse battery staple" not in users_file.read_text()
    assert raw_store["users"]["alice"]["password_hash"].startswith("pbkdf2_sha256$")

    assert authenticate_user(users_file, "alice", "wrong password") is None
    authenticated = authenticate_user(
        users_file,
        "alice",
        "correct horse battery staple",
    )

    assert authenticated == added

    assert remove_user(users_file, "alice") is True
    assert list_users(users_file) == []
    assert authenticate_user(users_file, "alice", "correct horse battery staple") is None


def test_user_store_rejects_duplicate_user_without_overwrite(tmp_path: Path) -> None:
    users_file = tmp_path / "users.json"
    add_user(users_file, username="alice", password="first password")

    with pytest.raises(UserStoreError, match="already exists"):
        add_user(users_file, username="alice", password="second password")


def test_user_store_can_overwrite_existing_user(tmp_path: Path) -> None:
    users_file = tmp_path / "users.json"
    add_user(users_file, username="alice", password="first password")

    add_user(
        users_file,
        username="alice",
        password="second password",
        display_name="Alice Updated",
        overwrite=True,
    )

    assert authenticate_user(users_file, "alice", "first password") is None
    authenticated = authenticate_user(users_file, "alice", "second password")
    assert authenticated is not None
    assert authenticated.display_name == "Alice Updated"


def test_resolve_users_file_prefers_explicit_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHAINLIT_AUTH_USERS_FILE", str(tmp_path / "env.json"))

    assert resolve_users_file(str(tmp_path / "explicit.json")) == tmp_path / "explicit.json"


def test_resolve_users_file_uses_env_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHAINLIT_AUTH_USERS_FILE", str(tmp_path / "env.json"))

    assert resolve_users_file(None) == tmp_path / "env.json"


def test_cli_users_add_list_and_remove(tmp_path: Path) -> None:
    users_file = tmp_path / "users.json"

    add_code = chainagents_cli.run_users_cli(
        [
            "--file",
            str(users_file),
            "add",
            "alice",
            "--display-name",
            "Alice Example",
            "--password",
            "correct horse battery staple",
        ],
        stdout=io.StringIO(),
        stderr=io.StringIO(),
        stdin=io.StringIO(""),
    )

    list_stdout = io.StringIO()
    list_code = chainagents_cli.run_users_cli(
        ["--file", str(users_file), "list", "--json"],
        stdout=list_stdout,
        stderr=io.StringIO(),
        stdin=io.StringIO(""),
    )

    remove_code = chainagents_cli.run_users_cli(
        ["--file", str(users_file), "remove", "alice"],
        stdout=io.StringIO(),
        stderr=io.StringIO(),
        stdin=io.StringIO(""),
    )

    assert add_code == 0
    assert list_code == 0
    assert remove_code == 0
    assert json.loads(list_stdout.getvalue()) == {
        "users": [{"display_name": "Alice Example", "username": "alice"}]
    }
    assert list_users(users_file) == []


def test_cli_users_json_flag_works_before_or_after_subcommand(tmp_path: Path) -> None:
    users_file = tmp_path / "users.json"
    add_user(users_file, username="alice", password="correct horse battery staple")

    before_stdout = io.StringIO()
    after_stdout = io.StringIO()

    before_code = chainagents_cli.run_users_cli(
        ["--file", str(users_file), "--json", "list"],
        stdout=before_stdout,
        stderr=io.StringIO(),
        stdin=io.StringIO(""),
    )
    after_code = chainagents_cli.run_users_cli(
        ["--file", str(users_file), "list", "--json"],
        stdout=after_stdout,
        stderr=io.StringIO(),
        stdin=io.StringIO(""),
    )

    assert before_code == 0
    assert after_code == 0
    assert json.loads(before_stdout.getvalue()) == json.loads(after_stdout.getvalue())


def test_cli_users_add_reads_password_from_stdin(tmp_path: Path) -> None:
    users_file = tmp_path / "users.json"

    code = chainagents_cli.run_users_cli(
        [
            "--file",
            str(users_file),
            "add",
            "alice",
            "--password-stdin",
        ],
        stdout=io.StringIO(),
        stderr=io.StringIO(),
        stdin=io.StringIO("correct horse battery staple\n"),
    )

    assert code == 0
    assert authenticate_user(users_file, "alice", "correct horse battery staple") is not None


def test_main_routes_users_command_before_runtime_startup(tmp_path: Path) -> None:
    users_file = tmp_path / "users.json"

    code = chainagents_cli.main(
        [
            "users",
            "--file",
            str(users_file),
            "add",
            "alice",
            "--password",
            "correct horse battery staple",
        ]
    )

    assert code == 0
    assert authenticate_user(users_file, "alice", "correct horse battery staple") is not None


def test_main_authenticates_users_from_configured_user_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    users_file = tmp_path / "users.json"
    add_user(
        users_file,
        username="alice",
        password="correct horse battery staple",
        display_name="Alice Example",
    )
    monkeypatch.setenv("CHAINLIT_AUTH_SECRET", "secret")
    monkeypatch.setenv("CHAINLIT_AUTH_USERS_FILE", str(users_file))
    monkeypatch.delenv("CHAINLIT_AUTH_USERNAME", raising=False)
    monkeypatch.delenv("CHAINLIT_AUTH_PASSWORD", raising=False)

    user = main.authenticate_configured_user(
        "alice",
        "correct horse battery staple",
    )

    assert user is not None
    assert user.identifier == "alice"
    assert user.display_name == "Alice Example"
    assert user.metadata == {"provider": "credentials", "source": "users_file"}


def test_main_preserves_legacy_single_user_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHAINLIT_AUTH_SECRET", "secret")
    monkeypatch.setenv("CHAINLIT_AUTH_USERNAME", "admin")
    monkeypatch.setenv("CHAINLIT_AUTH_PASSWORD", "change-me")
    monkeypatch.delenv("CHAINLIT_AUTH_USERS_FILE", raising=False)

    user = main.authenticate_configured_user("admin", "change-me")

    assert user is not None
    assert user.identifier == "admin"
    assert user.metadata == {"provider": "credentials", "source": "env"}


def test_main_authentication_is_disabled_without_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    users_file = tmp_path / "users.json"
    add_user(users_file, username="alice", password="correct horse battery staple")
    monkeypatch.delenv("CHAINLIT_AUTH_SECRET", raising=False)
    monkeypatch.setenv("CHAINLIT_AUTH_USERS_FILE", str(users_file))

    assert main.auth_enabled_from_env() is False
    assert main.authenticate_configured_user(
        "alice",
        "correct horse battery staple",
    ) is None
