"""Base module for LLM client abstraction.

LLMクライアントの抽象化を定義するモジュール.
"""

from __future__ import annotations

from typing import Protocol


class LLMError(Exception):
    """Base exception for LLM-related errors.

    LLM関連エラーの基底例外クラス.
    """

    pass


class LLMTimeoutError(LLMError):
    """Exception raised when LLM request times out.

    LLMリクエストがタイムアウトした場合に発生する例外.
    """

    pass


class LLMAPIError(LLMError):
    """Exception raised when LLM API returns an error.

    LLM APIがエラーを返した場合に発生する例外.
    """

    pass


class LLMClient(Protocol):
    """Protocol for LLM client implementations.

    LLMクライアント実装のためのプロトコル.
    """

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a response from the LLM.

        LLMから応答を生成する.

        Args:
            system_prompt (str): System prompt for the LLM / LLMへのシステムプロンプト
            user_prompt (str): User prompt for the LLM / LLMへのユーザープロンプト

        Returns:
            str: Generated response / 生成された応答

        Raises:
            LLMError: If an error occurs during generation / 生成中にエラーが発生した場合
            LLMTimeoutError: If the request times out / リクエストがタイムアウトした場合
        """
        ...
