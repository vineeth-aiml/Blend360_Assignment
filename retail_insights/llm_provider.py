"""
Gemini LLM provider adapter for the Retail Insights Assistant.

Uses Google Gemini through google-genai and protects the app from temporary
503 errors and quota/rate-limit failures. 429 quota errors are NOT retried in a
loop because that makes the UI feel stuck and can waste calls.
"""
from __future__ import annotations

import hashlib
import logging
import os
import random
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Iterable, Union

from google import genai
from google.genai import types
from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)


class GeminiQuotaError(RuntimeError):
    """Raised when Gemini reports a quota/rate-limit/API-key problem."""


@dataclass
class GeminiLLM:
    """Small adapter exposing a LangChain-like invoke API backed by Gemini."""

    api_key: str
    model: str = "gemini-2.5-flash"
    temperature: float = 0.1
    max_retries: int = 2
    fallback_model: str | None = None
    _cache: dict[str, str] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required to initialize GeminiLLM")

        self.api_key = self.api_key.strip()
        self.client = genai.Client(api_key=self.api_key)
        self.fallback_model = (
            self.fallback_model
            or os.getenv("GEMINI_FALLBACK_MODEL")
            or "gemini-2.5-flash-lite"
        ).strip()

    @staticmethod
    def _message_role(message: BaseMessage) -> str:
        msg_type = getattr(message, "type", "human")
        if msg_type == "system":
            return "SYSTEM"
        if msg_type in {"ai", "assistant"}:
            return "ASSISTANT"
        return "USER"

    def _format_messages(self, messages: Union[Iterable[BaseMessage], str]) -> str:
        if isinstance(messages, str):
            return messages

        formatted_parts: list[str] = []
        for message in messages:
            content = getattr(message, "content", str(message))
            role = self._message_role(message)
            formatted_parts.append(f"{role}:\n{content}")
        return "\n\n".join(formatted_parts)

    @staticmethod
    def _error_text(error: Exception) -> str:
        return str(error).upper()

    @classmethod
    def _is_quota_or_key_error(cls, error: Exception) -> bool:
        text = cls._error_text(error)
        return any(
            marker in text
            for marker in (
                "429",
                "RESOURCE_EXHAUSTED",
                "QUOTA",
                "API_KEY_INVALID",
                "API KEY EXPIRED",
                "BILLING",
            )
        )

    @classmethod
    def _is_retryable_server_error(cls, error: Exception) -> bool:
        text = cls._error_text(error)
        return any(marker in text for marker in ("503", "UNAVAILABLE", "500", "INTERNAL"))

    def _generate_once(self, prompt: str, model_name: str) -> str:
        response = self.client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=self.temperature),
        )
        return response.text or ""

    def invoke(self, messages: Union[Iterable[BaseMessage], str], **_: Any) -> SimpleNamespace:
        prompt = self._format_messages(messages)
        cache_key = hashlib.sha256(f"{self.model}|{self.temperature}|{prompt}".encode("utf-8")).hexdigest()
        if cache_key in self._cache:
            return SimpleNamespace(content=self._cache[cache_key])

        models_to_try = [self.model]
        if self.fallback_model and self.fallback_model != self.model:
            models_to_try.append(self.fallback_model)

        last_error: Exception | None = None
        quota_errors: list[str] = []

        for model_name in models_to_try:
            for attempt in range(self.max_retries + 1):
                try:
                    content = self._generate_once(prompt, model_name)
                    self._cache[cache_key] = content
                    return SimpleNamespace(content=content)

                except Exception as error:
                    last_error = error

                    # 429/quota/key errors should not be retried repeatedly.
                    # Try the fallback model once, then return a clear app-level error.
                    if self._is_quota_or_key_error(error):
                        quota_errors.append(f"{model_name}: {error}")
                        logger.warning("Gemini quota/key error on %s: %s", model_name, error)
                        break

                    if not self._is_retryable_server_error(error):
                        raise

                    if attempt >= self.max_retries:
                        logger.warning(
                            "Gemini model %s failed after server-error retries: %s",
                            model_name,
                            error,
                        )
                        break

                    delay = min((2 ** attempt) + random.uniform(0, 1), 6)
                    logger.warning(
                        "Gemini temporary server error on %s. Retry %s/%s in %.1fs: %s",
                        model_name,
                        attempt + 1,
                        self.max_retries,
                        delay,
                        error,
                    )
                    time.sleep(delay)

        if quota_errors:
            raise GeminiQuotaError(
                "Gemini quota/rate limit reached or API key invalid. "
                "Use a fresh key, wait for quota reset, or enable billing. "
                f"Details: {quota_errors[-1]}"
            )

        raise last_error or RuntimeError("Gemini generation failed")
