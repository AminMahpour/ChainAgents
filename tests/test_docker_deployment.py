from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (PROJECT_ROOT / path).read_text()


def _block_after(text: str, header: str, *, indent: int) -> str:
    target = f"{' ' * indent}{header}:"
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line == target or line.startswith(f"{target} "):
            block: list[str] = []
            for child in lines[index + 1 :]:
                stripped = child.strip()
                child_indent = len(child) - len(child.lstrip(" "))
                if stripped and child_indent <= indent:
                    break
                block.append(child)
            return "\n".join(block)
    return ""


def test_compose_defines_default_chainagents_app_service() -> None:
    compose = _read("compose.yaml")

    app = _block_after(compose, "app", indent=2)

    assert app
    assert "build:" in app
    assert "context: ." in app
    assert '"${CHAINAGENTS_PORT:-8000}:8000"' in app
    assert "DEEPAGENT_MODEL_BASE_URL: ${DEEPAGENT_MODEL_BASE_URL:-http://host.docker.internal:11434}" in app
    assert "DATABASE_URL: ${DATABASE_URL-postgresql://chainagents:chainagents@postgres:5432/chainagents?sslmode=disable}" in app
    assert "CHAINLIT_AUTH_USERS_FILE: ${CHAINLIT_AUTH_USERS_FILE:-/app/.files/users.json}" in app
    assert "depends_on:" in app
    assert "postgres:" in app
    assert "condition: service_healthy" in app
    assert "chainagents-files:/app/.files" in app
    assert "chainagents-rag:/app/.rag" in app


def test_compose_starts_postgres_with_app_service() -> None:
    compose = _read("compose.yaml")

    app = _block_after(compose, "app", indent=2)
    postgres = _block_after(compose, "postgres", indent=2)

    assert postgres
    assert "depends_on:" in app
    assert "postgres:" in app
    assert "profiles:" not in postgres
    assert "POSTGRES_USER: chainagents" in postgres
    assert "pg_isready -U chainagents -d chainagents" in postgres
    assert "postgres-data:/var/lib/postgresql/data" in postgres


def test_dockerfile_launches_chainlit_on_container_interface() -> None:
    dockerfile = _read("Dockerfile")

    assert "ghcr.io/astral-sh/uv:python3.12-bookworm-slim" in dockerfile
    assert "nodejs" in dockerfile
    assert "npm" in dockerfile
    assert "EXPOSE 8000" in dockerfile
    assert '"chainlit"' in dockerfile
    assert '"run"' in dockerfile
    assert '"main.py"' in dockerfile
    assert '"--host"' in dockerfile
    assert '"0.0.0.0"' in dockerfile
    assert '"--port"' in dockerfile
    assert '"8000"' in dockerfile


def test_dockerignore_excludes_local_runtime_state() -> None:
    dockerignore = _read(".dockerignore").splitlines()

    ignored = {line.strip() for line in dockerignore if line.strip() and not line.startswith("#")}

    assert ".git" in ignored
    assert ".venv" in ignored
    assert ".files" in ignored
    assert ".rag" in ignored
    assert ".pytest_cache" in ignored
    assert "__pycache__" in ignored
    assert "ChainAgents.egg-info" in ignored
