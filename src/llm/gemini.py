"""Gemini LLM client implementation.

Gemini LLMクライアントの実装モジュール.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

from llm.base import LLMAPIError, LLMError, LLMTimeoutError

logger = logging.getLogger(__name__)


class GeminiClient:
    """Gemini API client implementation.

    Gemini APIクライアントの実装.
    """

    def __init__(
        self,
        config: dict[str, Any],
    ) -> None:
        """Initialize the Gemini client.

        Geminiクライアントを初期化する.

        Args:
            config (dict[str, Any]): Configuration dictionary containing LLM settings /
                                     LLM設定を含む設定辞書
        """
        load_dotenv()

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise LLMError("GOOGLE_API_KEY environment variable is not set")

        self.client = genai.Client(api_key=api_key)
        self.config = config

        llm_config = config.get("llm", {})
        self.model = str(llm_config.get("model", "gemini-2.5-flash-lite"))
        self.timeout = int(llm_config.get("timeout", 10))
        self.max_retries = int(llm_config.get("max_retries", 2))

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a response from Gemini.

        Geminiから応答を生成する.

        Args:
            system_prompt (str): System prompt for the LLM / LLMへのシステムプロンプト
            user_prompt (str): User prompt for the LLM / LLMへのユーザープロンプト

        Returns:
            str: Generated response / 生成された応答

        Raises:
            LLMError: If an error occurs during generation / 生成中にエラーが発生した場合
            LLMTimeoutError: If the request times out / リクエストがタイムアウトした場合
        """
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                return self._generate_with_timeout(system_prompt, user_prompt)
            except LLMTimeoutError:
                last_error = LLMTimeoutError(
                    f"Request timed out after {self.timeout} seconds",
                )
                logger.warning(
                    "LLM request timed out (attempt %d/%d)",
                    attempt + 1,
                    self.max_retries + 1,
                )
            except LLMAPIError as e:
                last_error = e
                logger.warning(
                    "LLM API error (attempt %d/%d): %s",
                    attempt + 1,
                    self.max_retries + 1,
                    e,
                )
            except Exception as e:  # noqa: BLE001
                last_error = LLMError(f"Unexpected error: {e}")
                logger.warning(
                    "Unexpected LLM error (attempt %d/%d): %s",
                    attempt + 1,
                    self.max_retries + 1,
                    e,
                )

            if attempt < self.max_retries:
                wait_time = 2**attempt
                logger.info("Retrying in %d seconds...", wait_time)
                time.sleep(wait_time)

        if last_error:
            raise last_error
        raise LLMError("Unknown error occurred")

    def _generate_with_timeout(self, system_prompt: str, user_prompt: str) -> str:
        """Generate response with timeout handling.

        タイムアウト処理付きで応答を生成する.

        Args:
            system_prompt (str): System prompt for the LLM / LLMへのシステムプロンプト
            user_prompt (str): User prompt for the LLM / LLMへのユーザープロンプト

        Returns:
            str: Generated response / 生成された応答

        Raises:
            LLMTimeoutError: If the request times out / リクエストがタイムアウトした場合
            LLMAPIError: If the API returns an error / APIがエラーを返した場合
        """
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    http_options=types.HttpOptions(
                        timeout=self.timeout * 1000,
                    ),
                ),
            )

            if response.text:
                return response.text.strip()
            raise LLMAPIError("Empty response from Gemini API")

        except TimeoutError as e:
            raise LLMTimeoutError(f"Request timed out: {e}") from e
        except Exception as e:
            if "timeout" in str(e).lower():
                raise LLMTimeoutError(f"Request timed out: {e}") from e
            raise LLMAPIError(f"API error: {e}") from e
