from typing import Any

from litellm import completion


class LiteLLMClient:
    """Minimal LiteLLM wrapper.

    Expected model format is LiteLLM-style, e.g.:
    - openai/gpt-4o-mini
    - gemini/gemini-2.0-flash
    """

    def __init__(self, api_key: str, base_url: str, model: str, provider: str | None = None) -> None:
        self.api_key = api_key
        self.api_base = (base_url or "").strip().rstrip("/")
        self.model = self._normalize_model_name(model=model, provider=provider)

    def create(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float,
        stream: bool = False,
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> Any:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
            "api_key": self.api_key,
        }

        if self.api_base:
            request["api_base"] = self.api_base

        if response_format is not None:
            request["response_format"] = response_format
        if tools is not None:
            request["tools"] = tools
        if tool_choice is not None:
            request["tool_choice"] = tool_choice

        return completion(**request)

    @staticmethod
    def _normalize_model_name(model: str, provider: str | None) -> str:
        normalized = model.strip()
        if not normalized:
            return "openai/gpt-4o-mini"

        normalized_provider = (provider or "").strip().lower()
        if normalized_provider and "/" not in normalized:
            return f"{normalized_provider}/{normalized}"

        return normalized
