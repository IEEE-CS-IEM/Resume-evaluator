import os
import sys
from dotenv import load_dotenv

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)

from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

import util.constants as constants

class LLMFactory:

    @staticmethod
    def get_model_name():
        load_dotenv(override=True)
        env_name = os.getenv("llm_provider", "")

        model_mapping = {
            "mistral": constants.mistral_llm,
            "gemini": constants.gemini_llm,
        }
        try:
            return model_mapping[env_name]
        except KeyError:
            raise ValueError(
                "Invalid environment name. Must be 'mistral', 'gemini', 'openai'."
            )

    @staticmethod
    def get_api_key():
        load_dotenv(override=True)
        env_name = os.getenv("llm_provider", "")

        api_key_mapping = {
            "mistral": os.getenv("mistral_api_key"),
            "gemini": os.getenv("gemini_api_key"),
        }

        try:
            return api_key_mapping[env_name]
        except KeyError:
            raise ValueError(
                "Invalid environment name. Must be 'mistral', 'gemini', or 'openai'."
            )

    @staticmethod
    def create_llm_instance(temperature=0.3, local_llm=False):
        if local_llm:
            return ChatOllama(
                model=constants.local_llm,  
                base_url=os.getenv("local_model_url"),  
            )
        model_name = LLMFactory.get_model_name()
        api_key = LLMFactory.get_api_key()

        if model_name == constants.mistral_llm:
            return ChatMistralAI(
                api_key=api_key, model_name=model_name, temperature=temperature
            )
        elif model_name == constants.gemini_llm:
            return ChatGoogleGenerativeAI(
                api_key=api_key, model=model_name, temperature=temperature
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")


    @staticmethod
    def invoke(
        system_prompt: str = None,
        human_message: str = None,
        temperature=0.3,
        local_llm=False
    ):
        
        llm = LLMFactory.create_llm_instance(temperature, local_llm)
        if not llm:
            raise ValueError("LLM instance could not be created.")
        """
        Human Message obtained from pipeline must have 2 sections 
        1. Current Resume Data : as obtained by the processed return of OCR (involving current skills, experience etc.)
        2. Job Description : as obtained given as input (by the user)
        """
        if system_prompt and human_message:
            system_prompt = system_prompt.replace("{", "{{").replace("}", "}}")
            human_message = human_message.replace("{", "{{").replace("}", "}}")
            prompt_obj = ChatPromptTemplate.from_messages(
                [
                    SystemMessagePromptTemplate.from_template(system_prompt),
                    HumanMessagePromptTemplate.from_template(human_message),
                ]
            )
            formatted_messages = prompt_obj.format_messages()
            return llm.invoke(formatted_messages)

        elif human_message:
            human_message = human_message.replace("{", "{{").replace("}", "}}")
            return llm.invoke(human_message)

        else:
            raise ValueError("Proper Input Required")

