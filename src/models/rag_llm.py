from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


@dataclass
class RagLLMConfig:
    model_name: str
    system_prompt: str
    max_output_tokens: int | None = 96
    reasoning_effort: str | None = "low"
    retry_on_incomplete: bool = True
    retry_max_output_tokens: int = 1024


@dataclass
class RagLLMResult:
    answer_text: str
    response_id: str = ""
    response_status: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    error_message: str = ""
    retry_count: int = 0
    final_max_output_tokens: int | None = None


def build_rag_prompt(question: str, evidence_passages: list[str], config: RagLLMConfig) -> list[dict[str, str]]:
    joined_evidence = "\n\n".join(
        f"Passage {index + 1}: {passage}" for index, passage in enumerate(evidence_passages)
    )
    user_prompt = f"Question: {question}\n\nEvidence:\n{joined_evidence}"
    return [
        {"role": "system", "content": config.system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_rag_input(question: str, evidence_passages: list[str]) -> str:
    joined_evidence = "\n\n".join(
        f"Passage {index + 1}:\n{passage}" for index, passage in enumerate(evidence_passages)
    )
    return f"Question:\n{question}\n\nEvidence:\n{joined_evidence}"


class ReferenceRagAnswerer:
    """Offline helper for validating RAG pipeline shape without API calls."""

    def generate_answer(self, question: str, evidence_passages: list[str], metadata: dict[str, Any]) -> RagLLMResult:
        _ = question
        _ = evidence_passages
        answer = _safe_text(metadata.get("supported_gold_answer"))
        if not answer:
            answer = "INSUFFICIENT_EVIDENCE"
        return RagLLMResult(answer_text=answer, response_status="completed")


class OpenAIRagAnswerer:
    def __init__(self, config: RagLLMConfig) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The 'openai' package is required for provider='openai'. Install it with "
                "'pip install openai' or rerun pip install -r requirements.txt'."
            ) from exc

        self._client = OpenAI()
        self._config = config

    def generate_answer(self, question: str, evidence_passages: list[str], metadata: dict[str, Any]) -> RagLLMResult:
        _ = metadata
        rag_input = build_rag_input(question, evidence_passages)
        response = self._create_response(rag_input, self._config.max_output_tokens)
        result = self._parse_response(response, retry_count=0, max_output_tokens=self._config.max_output_tokens)

        if self._should_retry(result):
            retry_max_output_tokens = max(
                self._config.retry_max_output_tokens,
                (self._config.max_output_tokens or 0) * 2,
            )
            retry_response = self._create_response(rag_input, retry_max_output_tokens)
            result = self._parse_response(
                retry_response,
                retry_count=1,
                max_output_tokens=retry_max_output_tokens,
            )

        return result

    def _create_response(self, rag_input: str, max_output_tokens: int | None):
        request: dict[str, Any] = {
            "model": self._config.model_name,
            "instructions": self._config.system_prompt,
            "input": rag_input,
            "text": {"format": {"type": "text"}},
        }
        if max_output_tokens is not None:
            request["max_output_tokens"] = max_output_tokens
        if self._config.reasoning_effort and self._model_supports_reasoning():
            request["reasoning"] = {"effort": self._config.reasoning_effort}

        return self._client.responses.create(**request)

    def _parse_response(self, response, retry_count: int, max_output_tokens: int | None) -> RagLLMResult:
        usage = getattr(response, "usage", None)
        response_status = str(getattr(response, "status", "") or "")
        output_text = (getattr(response, "output_text", "") or "").strip()
        error_message = ""
        if response_status and response_status != "completed":
            error_message = f"OpenAI response status was {response_status}"

        return RagLLMResult(
            answer_text=output_text,
            response_id=str(getattr(response, "id", "") or ""),
            response_status=response_status,
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
            error_message=error_message,
            retry_count=retry_count,
            final_max_output_tokens=max_output_tokens,
        )

    def _should_retry(self, result: RagLLMResult) -> bool:
        if not self._config.retry_on_incomplete:
            return False
        if result.retry_count > 0:
            return False
        if result.response_status and result.response_status != "completed":
            return True
        return not result.answer_text

    def _model_supports_reasoning(self) -> bool:
        model_name = self._config.model_name.strip().lower()
        return model_name.startswith(("gpt-5", "o"))


def create_rag_answerer(provider: str, config: RagLLMConfig):
    provider_name = provider.strip().lower()
    if provider_name == "reference":
        return ReferenceRagAnswerer()
    if provider_name == "openai":
        return OpenAIRagAnswerer(config)
    raise ValueError(f"Unsupported RAG provider: {provider}")


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()
