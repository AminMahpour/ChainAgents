"""Build and parse command-line arguments for the ChainAgents CLI."""

from __future__ import annotations

import argparse

from chainagents.runtime import normalize_model_provider

DEFAULT_CLI_THREAD_ID = "cli"


def parse_model_provider_argument(value: str) -> str:
    """Normalize existing provider aliases while keeping Cortex canonical."""
    candidate = value.strip()
    if candidate.lower() == "snowflake_cortex" and candidate != "snowflake_cortex":
        raise argparse.ArgumentTypeError(
            "Snowflake Cortex must use the exact provider value 'snowflake_cortex'."
        )
    try:
        return normalize_model_provider(candidate)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    """Build parser.

    Returns:
        The constructed parser.
    """
    parser = argparse.ArgumentParser(
        prog="chainagents",
        description="Run the ChainAgents DeepAgent runtime without the Chainlit UI.",
    )
    parser.add_argument("prompt_parts", nargs="*", metavar="PROMPT")
    parser.add_argument("--prompt", help="Prompt to send to the agent.")
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read the prompt from stdin.",
    )
    parser.add_argument(
        "--photo",
        action="append",
        default=[],
        metavar="PATH",
        help="Attach an image file to the prompt for vision-capable models. May be repeated.",
    )

    parser.add_argument("--config", help="Path to deepagent.toml.")
    parser.add_argument(
        "--configure",
        action="store_true",
        help="Interactively configure deepagent.toml and exit.",
    )
    parser.add_argument("--database-url", help="Postgres URL for durable state.")
    parser.add_argument(
        "--no-database",
        action="store_true",
        help="Force in-memory state even when DATABASE_URL is set.",
    )
    parser.add_argument(
        "--provider",
        type=parse_model_provider_argument,
        metavar="PROVIDER",
        help=(
            "Model provider: ollama, openai_compatible, Snowflake Cortex "
            "(`snowflake_cortex`), anthropic, or claude."
        ),
    )
    parser.add_argument("--base-url", help="Model server base URL.")
    parser.add_argument(
        "--endpoint-url",
        help=(
            "Full model endpoint URL. Use this for non-standard "
            "/chat/completions, /responses, or Anthropic /v1/messages paths."
        ),
    )
    parser.add_argument("--model", help="Model name to run.")
    parser.add_argument("--api-key", help="API key for OpenAI-compatible model servers.")
    parser.add_argument("--temperature", type=float, help="Model temperature.")
    parser.add_argument(
        "--disable-streaming-for-tool-calls",
        action="store_true",
        help=(
            "Bypass model streaming only for requests that include tools. "
            "Useful for model servers with unreliable streamed tool-call chunks."
        ),
    )
    parser.add_argument(
        "--reasoning",
        choices=("low", "medium", "high"),
        help="Reasoning effort for this run.",
    )
    parser.add_argument(
        "--thread-id",
        default=None,
        help=(
            f"LangGraph thread ID. Defaults to {DEFAULT_CLI_THREAD_ID!r}, "
            "or 'tui' in --tui mode."
        ),
    )
    parser.add_argument("--recursion-limit", type=int, help="LangGraph recursion limit.")

    parser.add_argument("--async-subagent-url", help="Override URL for async subagents.")
    parser.add_argument("--mcp-session-id", help="Session scope for stateful MCP servers.")
    parser.add_argument(
        "--command",
        help="Run a configured command using the prompt as command input.",
    )

    parser.add_argument(
        "--rebuild-rag",
        action="store_true",
        help="Rebuild the configured workspace documentation RAG index.",
    )
    parser.add_argument(
        "--upload-rag",
        action="append",
        default=[],
        metavar="PATH",
        help="Add a file to the current thread's uploaded RAG index.",
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Disable RAG for this CLI process.",
    )

    parser.add_argument(
        "--status",
        action="store_true",
        help="Print resolved runtime status.",
    )
    parser.add_argument(
        "--list-commands",
        action="store_true",
        help="List configured commands and exit unless a prompt is also provided.",
    )
    parser.add_argument(
        "--stream",
        dest="stream",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stream the final response as it is produced.",
    )
    parser.add_argument(
        "--show-reasoning",
        action="store_true",
        help="Print streamed reasoning traces to stderr.",
    )
    parser.add_argument(
        "--show-tools",
        action="store_true",
        help="Print tool call traces to stderr.",
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Print machine-readable JSON output.",
    )
    parser.add_argument(
        "--tui",
        action="store_true",
        help="Run the full-screen terminal UI.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse args.

    Args:
        argv: The argv value.

    Returns:
        The parsed args.
    """
    return build_parser().parse_args(argv)
