import instructor             # enforces structured outputs from LLMs
from . import PromptTemplates as pt # provides prompt templates for a range of LLM tasks
from openai import OpenAI
import ollama
from pydantic import BaseModel, Field
from typing import List, Optional
import requests


# response structure for requests to get keywords
class KeywordList(BaseModel):
    keywords: List[str] = Field(
        ..., description="List of relevant semantic keywords or keyphrases, most relevant first"
    )

# response structure for requests to get a summary
class Summary(BaseModel):
    summary: str = Field(..., description="A factual, concise summary in passive voice under 25 words")

class LLMService:

    # default to a phi3:mini model served by locally hosted Ollama inference server on its standard port
    def __init__(self, provider: str = 'ollama', model: str = 'phi3:mini', server: str = 'localhost', port: int = 11434):

        self.provider = provider
        self.model = model
        self.server = server
        self.port = port

        self._get_client(provider, server, port)
        self._set_model(model)

        if self._validate_model(provider, model):
            self.client = self._get_client(provider, server, port)


    def _get_client(self, provider: str = None, server: str = None, port: int = None):
        if provider == 'ollama':
            ### deprecated method (new method allows for remote ollama and change of port)
            ### self.client = instructor.from_provider(f"ollama/{model}")
            try:
                self.client = instructor.from_openai(
                    OpenAI(
                        base_url=f"http://{server}:{port}/v1",
                        api_key='ollama',  # required, but unused
                    ),
                    mode=instructor.Mode.JSON,
                )
            except Exception as e:
                raise Exception(f"[Ollama Model Instantiation Error]: {e}")
        else:
            raise ValueError(f"{provider} is not supported as a provider. "
                             "Currently, only 'ollama' is supported. "
                             "However, Bedrock support is on the roadmap")
        

    def _set_model(self, model: str = None):
        
        if model is None:
            raise ValueError(f"No model specified for LLMService._get_client(). "
                              "Creation of an Ollama client requires a model to be specified.")


    def _available_models(self, provider: str = None) -> List[str]:
        """
        return a list of all models available on the Ollama server
        container compatible version - uses REST API rather than localhost only "ollama list" 
        Raises exceptions if the Ollama server isn't running or raises an exception.
        """
        if provider == 'ollama':
            available_models = []

            server_address = f"http://{self.server}:{self.port}/"
            url = f"{server_address}/api/tags"
            try:
                response = requests.get(url, timeout=5)
                response.raise_for_status()
                data = response.json()
                if data and 'models' in data:
                    for model in data['models']:
                        available_models.append(model.get('name', 'N/A'))
            except requests.exceptions.ConnectionError as e:
                raise ConnectionError(f"Could not connect to Ollama server at {server_address}. "
                                      "It might not be running or is on a different address/port.\n"
                                      "Please ensure Ollama is running (e.g. by running 'ollama serve' in your terminal).")
            except requests.exceptions.HTTPError as e:
                raise Exception(f"The request for {url} failed with a response code of {e.response.status_code}")
            except requests.exceptions.RequestException as e:
                raise Exception(f"Unexpected failure when requesting available models from Ollama: {e}")
            return available_models
        else:
            print(f"retrieval of available models is only supported for ollama - skipping this step for {provider}")
            return None


    def _validate_model(self, provider: str = None, model: str = None) -> bool:
        if provider is None:
            raise ValueError("LLM provider is undefined. Please set LLM_PROVIDER in your .env file.")
            #print(f"_validate_model: unspecified provider.")
            #return False
        if model is None:
            raise ValueError("LLM MODEL is undefined. Please set LLM_MODEL in your .env file.")
            #print(f"_validate_model: unspecified model.")
            #return False
        if provider == 'ollama':
            available_models = self._available_models(provider)
            if not model in available_models:
                print(  f"{model} is not available as a model on the ollama server.\n"
                        f"Please change OLLAMA_MODEL to an available model in your .env file "
                        f"or download it by typing 'ollama pull {model}' in the terminal.\n"
                        f"Currently available models are {available_models}."   )
            return False
        else:
            print(f"model validation is not yet supported for {provider}. skipping...")
        return True


    def get_keywords(self, text: str, top_n: int = 10, retries: int = 3) -> List[str]:
        if not text or not isinstance(text, str):
            return []
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                response_model=KeywordList,
                messages=[{"role": "user", 
                           "content": pt.PromptTemplates.get_keywords(text, top_n)}]
            )
            keywords = response.keywords
            if len(keywords) > top_n:
                print(f"[LLM ignored max count (top_n), trimming from {len(keywords)} to {top_n}]")
            return keywords[:top_n]
        except Exception as e:
            print(f"[Keyword Extraction Error] {e}")
            return []


    def get_summary(self, text: str, max_length: int = 25, retries: int = 3) -> Optional[str]:
        if not text or not isinstance(text, str):
            return None
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                response_model=Summary,
                messages=[{"role": "user", 
                           "content": pt.PromptTemplates.get_summary(text)}],
                max_retries=retries
            )
            return response.summary
        except Exception as e:
            print(f"[Summary Generation Error] {e}")
            return None
        

