"""Chainlit data-layer helpers for local Postgres persistence."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from chainlit.data.chainlit_data_layer import ChainlitDataLayer
from chainlit.data.storage_clients.base import BaseStorageClient
from chainlit.logger import logger

CHAINLIT_SCHEMA_BOOTSTRAP_ENV = "CHAINLIT_SCHEMA_BOOTSTRAP"

CHAINLIT_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS "User" (
        id TEXT PRIMARY KEY,
        identifier TEXT NOT NULL UNIQUE,
        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
        "createdAt" TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        "updatedAt" TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS "Thread" (
        id TEXT PRIMARY KEY,
        "createdAt" TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        "updatedAt" TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        "deletedAt" TIMESTAMPTZ,
        name TEXT,
        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
        tags TEXT[] NOT NULL DEFAULT '{}'::text[],
        "userId" TEXT REFERENCES "User"(id) ON DELETE SET NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS "Step" (
        id TEXT PRIMARY KEY,
        "createdAt" TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        "updatedAt" TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        "parentId" TEXT REFERENCES "Step"(id) ON DELETE CASCADE,
        "threadId" TEXT REFERENCES "Thread"(id) ON DELETE CASCADE,
        input TEXT,
        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
        name TEXT,
        output TEXT,
        type TEXT NOT NULL DEFAULT 'undefined',
        "showInput" TEXT DEFAULT 'json',
        "isError" BOOLEAN DEFAULT FALSE,
        "startTime" TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        "endTime" TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS "Feedback" (
        id TEXT PRIMARY KEY,
        "createdAt" TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        "updatedAt" TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        "stepId" TEXT REFERENCES "Step"(id) ON DELETE SET NULL,
        name TEXT NOT NULL,
        value DOUBLE PRECISION NOT NULL,
        comment TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS "Element" (
        id TEXT PRIMARY KEY,
        "createdAt" TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        "updatedAt" TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        "threadId" TEXT REFERENCES "Thread"(id) ON DELETE CASCADE,
        "stepId" TEXT NOT NULL REFERENCES "Step"(id) ON DELETE CASCADE,
        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
        mime TEXT,
        name TEXT,
        "objectKey" TEXT,
        url TEXT,
        "chainlitKey" TEXT,
        display TEXT,
        size TEXT,
        language TEXT,
        page INTEGER,
        props JSONB
    )
    """,
    'CREATE INDEX IF NOT EXISTS "User_identifier_idx" ON "User" (identifier)',
    'CREATE INDEX IF NOT EXISTS "Thread_createdAt_idx" ON "Thread" ("createdAt")',
    'CREATE INDEX IF NOT EXISTS "Thread_name_idx" ON "Thread" (name)',
    'CREATE INDEX IF NOT EXISTS "Thread_userId_idx" ON "Thread" ("userId")',
    'CREATE INDEX IF NOT EXISTS "Thread_updatedAt_idx" ON "Thread" ("updatedAt")',
    'CREATE INDEX IF NOT EXISTS "Step_createdAt_idx" ON "Step" ("createdAt")',
    'CREATE INDEX IF NOT EXISTS "Step_endTime_idx" ON "Step" ("endTime")',
    'CREATE INDEX IF NOT EXISTS "Step_parentId_idx" ON "Step" ("parentId")',
    'CREATE INDEX IF NOT EXISTS "Step_startTime_idx" ON "Step" ("startTime")',
    'CREATE INDEX IF NOT EXISTS "Step_threadId_idx" ON "Step" ("threadId")',
    'CREATE INDEX IF NOT EXISTS "Step_type_idx" ON "Step" (type)',
    'CREATE INDEX IF NOT EXISTS "Step_name_idx" ON "Step" (name)',
    'CREATE INDEX IF NOT EXISTS "Step_thread_start_end_idx" ON "Step" ("threadId", "startTime", "endTime")',
    'CREATE INDEX IF NOT EXISTS "Feedback_createdAt_idx" ON "Feedback" ("createdAt")',
    'CREATE INDEX IF NOT EXISTS "Feedback_name_idx" ON "Feedback" (name)',
    'CREATE INDEX IF NOT EXISTS "Feedback_stepId_idx" ON "Feedback" ("stepId")',
    'CREATE INDEX IF NOT EXISTS "Feedback_value_idx" ON "Feedback" (value)',
    'CREATE INDEX IF NOT EXISTS "Feedback_name_value_idx" ON "Feedback" (name, value)',
    'CREATE INDEX IF NOT EXISTS "Element_stepId_idx" ON "Element" ("stepId")',
    'CREATE INDEX IF NOT EXISTS "Element_threadId_idx" ON "Element" ("threadId")',
)


def chainlit_schema_bootstrap_enabled() -> bool:
    """Return whether this app should create missing Chainlit tables at startup."""
    raw_value = os.getenv(CHAINLIT_SCHEMA_BOOTSTRAP_ENV, "true").strip().lower()
    return raw_value not in {"0", "false", "no", "off"}


class AutoMigratingChainlitDataLayer(ChainlitDataLayer):
    """Chainlit official asyncpg data layer with idempotent schema setup."""

    def __init__(
        self,
        database_url: str,
        storage_client: BaseStorageClient | None = None,
        show_logger: bool = False,
        *,
        bootstrap_schema: bool = True,
    ) -> None:
        super().__init__(
            database_url=database_url,
            storage_client=storage_client,
            show_logger=show_logger,
        )
        self.bootstrap_schema = bootstrap_schema
        self._schema_ready = False
        self._schema_lock = asyncio.Lock()

    async def connect(self) -> None:
        """Open the connection pool and create Chainlit tables before first query."""
        await super().connect()
        if self.bootstrap_schema:
            await self.setup_schema()

    async def setup_schema(self) -> None:
        """Create the official Chainlit Postgres tables and indexes if absent."""
        if self._schema_ready:
            return
        async with self._schema_lock:
            if self._schema_ready:
                return
            if not self.pool:
                raise RuntimeError("Chainlit data layer pool is not initialized")
            async with self.pool.acquire() as connection:  # type: ignore[union-attr]
                async with connection.transaction():
                    for statement in CHAINLIT_SCHEMA_STATEMENTS:
                        await connection.execute(statement)
            self._schema_ready = True
            logger.info("Chainlit Postgres schema is ready")


def create_chainlit_storage_client() -> BaseStorageClient | None:
    """Build the optional Chainlit element storage client from environment variables."""
    bucket_name = os.getenv("BUCKET_NAME")

    aws_region = os.getenv("APP_AWS_REGION")
    aws_access_key = os.getenv("APP_AWS_ACCESS_KEY")
    aws_secret_key = os.getenv("APP_AWS_SECRET_KEY")
    dev_aws_endpoint = os.getenv("DEV_AWS_ENDPOINT")
    is_using_s3 = bool(aws_access_key and aws_secret_key and aws_region)

    gcs_project_id = os.getenv("APP_GCS_PROJECT_ID")
    gcs_client_email = os.getenv("APP_GCS_CLIENT_EMAIL")
    gcs_private_key = os.getenv("APP_GCS_PRIVATE_KEY")
    is_using_gcs = bool(gcs_project_id)

    azure_storage_account = os.getenv("APP_AZURE_STORAGE_ACCOUNT")
    azure_storage_key = os.getenv("APP_AZURE_STORAGE_ACCESS_KEY")
    is_using_azure = bool(azure_storage_account and azure_storage_key)

    configured_storage_count = sum([is_using_s3, is_using_gcs, is_using_azure])
    if configured_storage_count > 1:
        logger.warning("Multiple Chainlit storage configurations detected; ignoring all")
        return None

    if is_using_s3:
        from chainlit.data.storage_clients.s3 import S3StorageClient

        return S3StorageClient(
            bucket=bucket_name,
            region_name=aws_region,
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            endpoint_url=dev_aws_endpoint,
        )

    if is_using_gcs:
        from chainlit.data.storage_clients.gcs import GCSStorageClient

        return GCSStorageClient(
            project_id=gcs_project_id,
            client_email=gcs_client_email,
            private_key=gcs_private_key,
            bucket_name=bucket_name,
        )

    if is_using_azure:
        from chainlit.data.storage_clients.azure_blob import AzureBlobStorageClient

        return AzureBlobStorageClient(
            container_name=bucket_name,
            storage_account=azure_storage_account,
            storage_key=azure_storage_key,
        )

    return None


def create_chainlit_data_layer(
    database_url: str | None = None,
) -> AutoMigratingChainlitDataLayer:
    """Create the app's Chainlit data layer for a configured database URL."""
    resolved_database_url = (database_url or os.getenv("DATABASE_URL", "")).strip()
    if not resolved_database_url:
        raise ValueError("DATABASE_URL is required to create the Chainlit data layer")
    return AutoMigratingChainlitDataLayer(
        database_url=resolved_database_url,
        storage_client=create_chainlit_storage_client(),
        bootstrap_schema=chainlit_schema_bootstrap_enabled(),
    )


def chainlit_data_layer_enabled(environ: dict[str, Any] | None = None) -> bool:
    """Return whether DATABASE_URL should enable the Chainlit data layer."""
    source = os.environ if environ is None else environ
    return bool(str(source.get("DATABASE_URL", "")).strip())
