"""Tests for Chainlit Postgres persistence bootstrap helpers."""

from __future__ import annotations

import pytest

from chainlit_persistence import (
    CHAINLIT_SCHEMA_STATEMENTS,
    AutoMigratingChainlitDataLayer,
    chainlit_data_layer_enabled,
    chainlit_schema_bootstrap_enabled,
)


class _FakeTransaction:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeConnection:
    def __init__(self) -> None:
        self.executed: list[str] = []

    def transaction(self) -> _FakeTransaction:
        return _FakeTransaction()

    async def execute(self, statement: str) -> None:
        self.executed.append(statement)


class _FakeAcquire:
    def __init__(self, connection: _FakeConnection) -> None:
        self.connection = connection

    async def __aenter__(self) -> _FakeConnection:
        return self.connection

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakePool:
    def __init__(self) -> None:
        self.connection = _FakeConnection()

    def acquire(self) -> _FakeAcquire:
        return _FakeAcquire(self.connection)


@pytest.mark.anyio
async def test_auto_migrating_data_layer_bootstraps_schema_once() -> None:
    """Verify schema bootstrap runs all DDL statements only once."""
    pool = _FakePool()
    data_layer = AutoMigratingChainlitDataLayer("postgresql://example/db")
    data_layer.pool = pool  # type: ignore[assignment]

    await data_layer.setup_schema()
    await data_layer.setup_schema()

    assert pool.connection.executed == list(CHAINLIT_SCHEMA_STATEMENTS)


def test_chainlit_schema_statements_create_official_tables() -> None:
    """Verify the bootstrap DDL includes the quoted tables used by Chainlit."""
    schema_sql = "\n".join(CHAINLIT_SCHEMA_STATEMENTS)

    for table_name in ('"User"', '"Thread"', '"Step"', '"Feedback"', '"Element"'):
        assert f"CREATE TABLE IF NOT EXISTS {table_name}" in schema_sql

    assert 'REFERENCES "User"(id)' in schema_sql
    assert 'REFERENCES "Thread"(id)' in schema_sql
    assert 'REFERENCES "Step"(id)' in schema_sql


def test_chainlit_data_layer_enabled_requires_database_url() -> None:
    """Verify Chainlit data layer registration follows DATABASE_URL."""
    assert chainlit_data_layer_enabled({"DATABASE_URL": "postgresql://host/db"})
    assert not chainlit_data_layer_enabled({"DATABASE_URL": ""})
    assert not chainlit_data_layer_enabled({})


@pytest.mark.parametrize("raw_value", ["0", "false", "False", "no", "off"])
def test_chainlit_schema_bootstrap_enabled_can_be_disabled(
    monkeypatch: pytest.MonkeyPatch,
    raw_value: str,
) -> None:
    """Verify operators can opt out when they run external Chainlit migrations."""
    monkeypatch.setenv("CHAINLIT_SCHEMA_BOOTSTRAP", raw_value)

    assert not chainlit_schema_bootstrap_enabled()


def test_chainlit_schema_bootstrap_enabled_defaults_to_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify missing-table errors are prevented by default."""
    monkeypatch.delenv("CHAINLIT_SCHEMA_BOOTSTRAP", raising=False)

    assert chainlit_schema_bootstrap_enabled()
