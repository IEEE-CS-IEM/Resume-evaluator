import os
import time
from typing import Dict, List, Optional, Sequence

import httpx
import requests
from dotenv import load_dotenv

from util import constants

load_dotenv()


class ProviderCapacityError(RuntimeError):
    """Raised when an upstream model rejects the request due to capacity."""
    pass


class MyAgent:
    def __init__(self,system_prompt):
        self.system_prompt = system_prompt
        self.provider = os.getenv("llm_provider", "gemini").strip().lower()
        fallback_env = (
            os.getenv("llm_provider_fallbacks")
            or os.getenv("LLM_PROVIDER_FALLBACKS")
            or ""
        )
        fallback_providers: List[str] = []
        for entry in fallback_env.split(","):
            candidate = entry.strip().lower()
            if not candidate or candidate == self.provider:
                continue
            if candidate not in fallback_providers:
                fallback_providers.append(candidate)
        self.fallback_providers = tuple(fallback_providers)

    def __call__(self,message,temperature=0.3):
        if not message or message.strip() == "":
            raise ValueError("Message cannot be empty")
        clean_message = message.strip()

        candidates: List[str] = []
        seen = set()
        for provider in (self.provider, *self.fallback_providers):
            lowered = provider.strip().lower()
            if not lowered or lowered in seen:
                continue
            seen.add(lowered)
            candidates.append(lowered)

        last_capacity_error: Optional[ProviderCapacityError] = None
        for index, provider in enumerate(candidates):
            try:
                return self._dispatch(provider, clean_message, temperature)
            except ProviderCapacityError as exc:
                last_capacity_error = exc
                continue
            except ValueError:
                # Preserve legacy behaviour for the primary provider,
                # but ignore misconfigured fallback providers.
                if index == 0:
                    raise
                continue

        if last_capacity_error is not None:
            raise last_capacity_error

        raise RuntimeError("No valid LLM provider available for MyAgent dispatch.")

    def _dispatch(self, provider: str, message: str, temperature: float):
        if provider == "mistral":
            return self._invoke_mistral(message, temperature)
        if provider == "gemini":
            return self._invoke_gemini(message, temperature)
        if provider == "local":
            return self._invoke_local(message, temperature)
        raise ValueError(
            f"Unsupported llm_provider '{provider}'. Expected 'gemini', 'mistral', or 'local'."
        )

    def _invoke_gemini(self, message: str, temperature: float):
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise RuntimeError(
                "google-generativeai is not installed. Install it to use the Gemini provider."
            ) from exc

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("gemini_api_key")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set in the environment.")

        model_name = (
            os.getenv("GEMINI_MODEL")
            or os.getenv("gemini_model")
            or constants.gemini_llm
        )

        # Force REST transport so newer models resolve correctly.
        genai.configure(api_key=api_key, transport="rest")
        client = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=self.system_prompt or None,
        )
        response = client.generate_content(
            message,
            generation_config={"temperature": temperature},
        )
        return getattr(response, "text", response)

    def _mistral_models(self) -> List[str]:
        primary = (
            os.getenv("MISTRAL_MODEL")
            or os.getenv("mistral_model")
            or constants.mistral_llm
        )
        fallback_env = (
            os.getenv("MISTRAL_MODEL_FALLBACKS")
            or os.getenv("mistral_model_fallbacks")
            or ""
        )
        models: List[str] = []
        seen = set()
        for candidate in [primary, *fallback_env.split(",")]:
            if not candidate:
                continue
            normalized = candidate.strip()
            if not normalized:
                continue
            lowered = normalized.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            models.append(normalized)
        return models

    def _mistral_backoff(self) -> float:
        raw = os.getenv("MISTRAL_RETRY_BACKOFF_SECONDS") or os.getenv(
            "mistral_retry_backoff_seconds"
        )
        if raw:
            try:
                value = float(raw)
                if value >= 0:
                    return value
            except ValueError:
                pass
        return 1.5

    def _mistral_retry_policy(self) -> tuple[int, float]:
        raw_max = os.getenv("MISTRAL_MAX_RETRIES") or os.getenv("mistral_max_retries")
        raw_multiplier = (
            os.getenv("MISTRAL_RETRY_MULTIPLIER")
            or os.getenv("mistral_retry_multiplier")
        )
        max_attempts = 3
        multiplier = 2.0
        if raw_max:
            try:
                candidate = int(raw_max)
                if candidate >= 1:
                    max_attempts = candidate
            except ValueError:
                pass
        if raw_multiplier:
            try:
                candidate = float(raw_multiplier)
                if candidate >= 1.0:
                    multiplier = candidate
            except ValueError:
                pass
        return max_attempts, multiplier

    def _invoke_mistral(self, message: str, temperature: float):
        try:
            from langchain_mistralai import ChatMistralAI
            from langchain_core.messages import HumanMessage, SystemMessage
        except ImportError as exc:
            raise RuntimeError(
                "langchain-mistralai (and langchain-core) must be installed for the Mistral provider."
            ) from exc

        api_key = os.getenv("MISTRAL_API_KEY") or os.getenv("mistral_api_key")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY is not set in the environment.")

        models = self._mistral_models()
        if not models:
            raise ValueError("No Mistral models configured.")

        backoff_seconds = self._mistral_backoff()
        max_attempts, multiplier = self._mistral_retry_policy()
        last_capacity_error: Optional[httpx.HTTPStatusError] = None

        for index, model_name in enumerate(models):
            client = ChatMistralAI(
                api_key=api_key,
                model=model_name,
                temperature=temperature,
            )
            messages = []
            if self.system_prompt:
                messages.append(SystemMessage(content=self.system_prompt))
            messages.append(HumanMessage(content=message))

            for attempt in range(1, max_attempts + 1):
                try:
                    response = client.invoke(messages)
                    return getattr(response, "content", response)
                except httpx.HTTPStatusError as exc:
                    if exc.response is not None and exc.response.status_code == 429:
                        last_capacity_error = exc
                        if attempt >= max_attempts:
                            break
                        delay = backoff_seconds * (multiplier ** (attempt - 1))
                        if delay > 0:
                            time.sleep(delay)
                        continue
                    raise

            # Optional cooldown before switching models if there are more to try.
            if index < len(models) - 1 and backoff_seconds:
                time.sleep(backoff_seconds)

        if last_capacity_error is not None:
            raise ProviderCapacityError("Mistral capacity exceeded") from last_capacity_error

        raise RuntimeError("Failed to invoke any configured Mistral model.")

    def _invoke_local(self, message: str, temperature: float):
        base_url = os.getenv("LOCAL_MODEL_URL") or os.getenv("local_model_url") or os.getenv("LOCAL_OLLAMA_URL")
        if not base_url:
            base_url = "http://localhost:11434"
        model_name = (
            os.getenv("LOCAL_MODEL")
            or os.getenv("local_model")
            or constants.local_llm
        )

        endpoint = base_url.rstrip("/") + "/api/chat"
        headers = {"Content-Type": "application/json"}
        payload: Dict[str, object] = {
            "model": model_name,
            "temperature": temperature,
            "messages": self._build_messages(message),
        }

        try:
            response = requests.post(endpoint, json=payload, timeout=120)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Failed to invoke local model '{model_name}' at {endpoint}: {exc}"
            ) from exc

        data = response.json()
        # Ollama returns either `message` or a list of messages.
        if isinstance(data, dict):
            if "message" in data:
                content = data["message"].get("content")
                if content:
                    return content
            if "messages" in data and isinstance(data["messages"], list):
                contents: List[str] = []
                for item in data["messages"]:
                    if isinstance(item, dict) and item.get("role") == "assistant":
                        text = item.get("content")
                        if isinstance(text, str):
                            contents.append(text)
                if contents:
                    return "\n".join(contents)

        raise RuntimeError(
            f"Unexpected response from local model '{model_name}': {data}"
        )

    def _build_messages(self, user_message: str) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_message})
        return messages


