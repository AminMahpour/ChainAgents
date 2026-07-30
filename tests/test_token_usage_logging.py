"""Test request-level token usage file logging."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from uuid import uuid4

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, Generation, LLMResult

from chainagents.runtime.token_usage import TokenUsageFileCallbackHandler


def usage_result(input_tokens: int, output_tokens: int) -> LLMResult:
    """Build a real LangChain result with normalized usage metadata."""
    return LLMResult(
        generations=[
            [
                ChatGeneration(
                    message=AIMessage(
                        content="done",
                        usage_metadata={
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "total_tokens": input_tokens + output_tokens,
                        },
                    )
                )
            ]
        ]
    )


def test_aggregates_model_calls_into_one_root_request_record(tmp_path: Path) -> None:
    """A missing accumulator or write-once boundary must fail this test."""
    log_path = tmp_path / ".files" / "token-usage.jsonl"
    handler = TokenUsageFileCallbackHandler(
        log_path=log_path,
    )
    handler.on_llm_end(usage_result(10, 4), run_id=uuid4())
    handler.on_llm_end(usage_result(6, 2), run_id=uuid4())
    root_id = uuid4()

    handler.on_chain_end({}, run_id=root_id)

    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert len(records) == 1
    record = records[0]
    assert isinstance(record.pop("timestamp"), str)
    assert record == {
        "request_id": str(root_id),
        "status": "success",
        "input_tokens": 16,
        "output_tokens": 6,
        "total_tokens": 22,
    }


def test_normalizes_provider_token_usage_when_standard_metadata_is_absent(
    tmp_path: Path,
) -> None:
    """Removing fallback normalization must make provider usage disappear."""
    log_path = tmp_path / "token-usage.jsonl"
    handler = TokenUsageFileCallbackHandler(
        log_path=log_path,
    )
    result = LLMResult(
        generations=[[Generation(text="done")]],
        llm_output={
            "token_usage": {
                "prompt_tokens": 5,
                "completion_tokens": 2,
            }
        },
    )

    handler.on_llm_end(result, run_id=uuid4())
    handler.on_chain_end({}, run_id=uuid4())

    record = json.loads(log_path.read_text())
    assert record["input_tokens"] == 5
    assert record["output_tokens"] == 2
    assert record["total_tokens"] == 7


def test_derives_total_when_provider_total_is_malformed(tmp_path: Path) -> None:
    """Trusting a malformed total must make the aggregate inconsistent."""
    log_path = tmp_path / "token-usage.jsonl"
    handler = TokenUsageFileCallbackHandler(
        log_path=log_path,
    )
    result = LLMResult(
        generations=[[Generation(text="done")]],
        llm_output={
            "token_usage": {
                "input_tokens": 5,
                "output_tokens": 2,
                "total_tokens": "7",
            }
        },
    )

    handler.on_llm_end(result, run_id=uuid4())
    handler.on_chain_end({}, run_id=uuid4())

    record = json.loads(log_path.read_text())
    assert record["input_tokens"] == 5
    assert record["output_tokens"] == 2
    assert record["total_tokens"] == 7


def test_standard_metadata_takes_precedence_over_provider_fallback(
    tmp_path: Path,
) -> None:
    """Counting both sources must inflate this request's token totals."""
    log_path = tmp_path / "token-usage.jsonl"
    result = usage_result(7, 3)
    result.llm_output = {
        "token_usage": {
            "prompt_tokens": 700,
            "completion_tokens": 300,
            "total_tokens": 1000,
        }
    }
    handler = TokenUsageFileCallbackHandler(
        log_path=log_path,
    )

    handler.on_llm_end(result, run_id=uuid4())
    handler.on_chain_end({}, run_id=uuid4())

    record = json.loads(log_path.read_text())
    assert record["input_tokens"] == 7
    assert record["output_tokens"] == 3
    assert record["total_tokens"] == 10


def test_nested_chain_completion_does_not_write_a_request_record(
    tmp_path: Path,
) -> None:
    """Removing the root-run check must create a premature nested record."""
    log_path = tmp_path / "token-usage.jsonl"
    handler = TokenUsageFileCallbackHandler(
        log_path=log_path,
    )

    handler.on_chain_end(
        {},
        run_id=uuid4(),
        parent_run_id=uuid4(),
    )

    assert not log_path.exists()


def test_repeated_root_terminal_events_write_once(tmp_path: Path) -> None:
    """Removing the terminal guard must duplicate the request record."""
    log_path = tmp_path / "token-usage.jsonl"
    handler = TokenUsageFileCallbackHandler(
        log_path=log_path,
    )
    root_id = uuid4()

    handler.on_chain_end({}, run_id=root_id)
    handler.on_chain_end({}, run_id=root_id)

    assert len(log_path.read_text().splitlines()) == 1


def test_root_error_writes_usage_accumulated_before_failure(tmp_path: Path) -> None:
    """Removing error finalization must lose tokens spent before a failure."""
    log_path = tmp_path / "token-usage.jsonl"
    handler = TokenUsageFileCallbackHandler(
        log_path=log_path,
    )
    handler.on_llm_end(usage_result(9, 3), run_id=uuid4())
    root_id = uuid4()

    handler.on_chain_error(RuntimeError("failed"), run_id=root_id)

    record = json.loads(log_path.read_text())
    assert record["request_id"] == str(root_id)
    assert record["status"] == "error"
    assert record["input_tokens"] == 9
    assert record["output_tokens"] == 3
    assert record["total_tokens"] == 12


def test_cancelled_root_writes_partial_usage_once(tmp_path: Path) -> None:
    """Cancellation must preserve completed model usage without duplicate records."""
    log_path = tmp_path / "token-usage.jsonl"
    handler = TokenUsageFileCallbackHandler(log_path=log_path)
    root_id = uuid4()
    handler.on_chain_start({}, {}, run_id=root_id)
    handler.on_llm_end(usage_result(9, 3), run_id=uuid4())

    handler.finalize_cancelled()
    handler.on_chain_end({}, run_id=root_id)

    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["request_id"] == str(root_id)
    assert records[0]["status"] == "cancelled"
    assert records[0]["input_tokens"] == 9
    assert records[0]["output_tokens"] == 3
    assert records[0]["total_tokens"] == 12


def test_model_error_includes_usage_from_partial_response(tmp_path: Path) -> None:
    """Dropping an error response must undercount tokens consumed by that call."""
    log_path = tmp_path / "token-usage.jsonl"
    handler = TokenUsageFileCallbackHandler(
        log_path=log_path,
    )

    handler.on_llm_error(
        RuntimeError("model failed"),
        run_id=uuid4(),
        response=usage_result(8, 1),
    )
    handler.on_chain_error(RuntimeError("run failed"), run_id=uuid4())

    record = json.loads(log_path.read_text())
    assert record["input_tokens"] == 8
    assert record["output_tokens"] == 1
    assert record["total_tokens"] == 9


def test_ignores_malformed_negative_and_boolean_token_counts(
    tmp_path: Path,
) -> None:
    """Accepting invalid counts must corrupt totals or raise during logging."""
    log_path = tmp_path / "token-usage.jsonl"
    handler = TokenUsageFileCallbackHandler(
        log_path=log_path,
    )
    result = LLMResult(
        generations=[[Generation(text="done")]],
        llm_output={
            "token_usage": {
                "input_tokens": True,
                "output_tokens": -3,
                "total_tokens": "10",
            }
        },
    )

    handler.on_llm_end(result, run_id=uuid4())
    handler.on_chain_end({}, run_id=uuid4())

    record = json.loads(log_path.read_text())
    assert record["input_tokens"] == 0
    assert record["output_tokens"] == 0
    assert record["total_tokens"] == 0


def test_file_write_failure_warns_without_failing_the_request(
    tmp_path: Path,
    caplog,
) -> None:
    """Propagating an OSError must fail this callback boundary test."""
    log_path = tmp_path / "token-usage.jsonl"
    log_path.mkdir()
    handler = TokenUsageFileCallbackHandler(
        log_path=log_path,
    )

    with caplog.at_level(logging.WARNING, logger="chainagents.runtime.token_usage"):
        handler.on_chain_end({}, run_id=uuid4())

    assert "Could not append token usage" in caplog.text
