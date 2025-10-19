import os
from dotenv import load_dotenv

from util import constants

load_dotenv()

class MyAgent:
    def __init__(self,system_prompt):
        self.system_prompt = system_prompt
        self.provider = os.getenv("llm_provider", "gemini").strip().lower()

    def __call__(self,message,temperature=0.3):
        if not message or message.strip() == "":
            raise ValueError("Message cannot be empty")
        clean_message = message.strip()

        if self.provider == "mistral":
            return self._invoke_mistral(clean_message, temperature)
        if self.provider == "gemini":
            return self._invoke_gemini(clean_message, temperature)
        raise ValueError(
            f"Unsupported llm_provider '{self.provider}'. Expected 'gemini' or 'mistral'."
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
            or f"{constants.gemini_llm}-latest"
        )

        genai.configure(api_key=api_key)
        client = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=self.system_prompt or None,
        )
        response = client.generate_content(
            message,
            generation_config={"temperature": temperature},
        )
        return getattr(response, "text", response)

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

        model_name = (
            os.getenv("MISTRAL_MODEL")
            or os.getenv("mistral_model")
            or constants.mistral_llm
        )

        client = ChatMistralAI(
            api_key=api_key,
            model=model_name,
            temperature=temperature,
        )
        messages = []
        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))
        messages.append(HumanMessage(content=message))

        response = client.invoke(messages)
        return getattr(response, "content", response)


