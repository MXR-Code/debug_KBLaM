from torch.nn import Module
from transformers import FeatureExtractionMixin
import sentence_transformers
import os
import sys
from pathlib import Path
from azure.identity import AuthenticationRecord
from azure.identity import DeviceCodeCredential
from azure.identity import TokenCachePersistenceOptions
from azure.identity import get_bearer_token_provider
from openai import AzureOpenAI


class GPT:
    def __init__(self,
                 model_name: str,
                 azure_endpoint_url: str,
                 api_version: str = "2024-02-15-preview",
                 system_msg: str = "You are an AI assistant.",
                 max_retries: int = 12,
                 temperature: int = 1.0,
                 max_tokens: int = 4096,
                 top_p: float = 0.95,
                 frequency_penalty: int = 0,
                 presence_penalty: int = 0,
                 seed: int = None, ):
        valid_models = ["gpt-4o", "ada-embeddings", "text-embedding-3-large"]
        if model_name not in valid_models:
            assert False

        azure_ad_token_provider = get_bearer_token_provider(self.get_credential(),
                                                            "https://cognitiveservices.azure.com/.default")

        self.OA_client = AzureOpenAI(azure_endpoint=azure_endpoint_url,
                                     api_version=api_version,
                                     azure_ad_token_provider=azure_ad_token_provider)

        self.max_retries = max_retries
        self.system_msg = system_msg
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.seed = seed

    def set_seed(self, seed: int):
        self.seed = seed

    def get_credential(self, lib_name: str = "azure_openai") -> DeviceCodeCredential:
        if sys.platform.startswith("win"):
            auth_record_root_path = Path(os.environ["LOCALAPPDATA"])
        else:
            auth_record_root_path = Path.home()

        auth_record_path = auth_record_root_path / lib_name / "auth_record.json"
        cache_options = TokenCachePersistenceOptions(name=f"{lib_name}.cache", allow_unencrypted_storage=True)

        if auth_record_path.exists():
            with open(auth_record_path, "r") as f:
                record_json = f.read()
            deserialized_record = AuthenticationRecord.deserialize(record_json)
            credential = DeviceCodeCredential(authentication_record=deserialized_record,
                                              cache_persistence_options=cache_options)
        else:
            auth_record_path.parent.mkdir(parents=True, exist_ok=True)
            credential = DeviceCodeCredential(cache_persistence_options=cache_options)
            record_json = credential.authenticate().serialize()
            with open(auth_record_path, "w") as f:
                f.write(record_json)

        return credential

    def api_call_chat(self, messages: list[dict]) -> str | None:
        for _ in range(self.max_retries):
            completion = self.OA_client.chat.completions.create(model=self.model_name,
                                                                messages=messages,
                                                                temperature=self.temperature,
                                                                max_tokens=self.max_tokens,
                                                                top_p=self.top_p,
                                                                frequency_penalty=self.frequency_penalty,
                                                                presence_penalty=self.presence_penalty,
                                                                seed=self.seed)
            if completion:
                return completion.choices[0].message.content
        return None

    def api_call_embedding(self, text: str) -> list[float] | None:
        for _ in range(self.max_retries):
            embedding = self.OA_client.embeddings.create(input=text, model=self.model_name)
            if embedding:
                return embedding.data[0].embedding
        return None

    def generate_response(self, prompt: str) -> str | None:
        messages = [{"role": "system", "content": self.system_msg},
                    {"role": "user", "content": prompt}]
        response = self.api_call_chat(messages)
        return response

    def generate_embedding(self, text: str) -> list[float] | None:
        embedding = self.api_call_embedding(text)
        return embedding


class SentenceEncoder(Module, FeatureExtractionMixin):
    kb_special_token = {"<KB_BEGIN>": 0,
                        "<KB_END>": 1,
                        "<KEY_SEP>": 2,
                        "<VALUE_SEP>": 3,
                        "<ENTITY_SEP>": 4,
                        "<KV_SEP>": 5}

    def __init__(self, model_name: str = None, azure_endpoint_url: str = None, device=None):
        super().__init__()
        self.model_name = model_name

        if model_name in ["text-embedding-3-large", "ada-embeddings"]:
            assert azure_endpoint_url
            self.model = GPT(model_name=model_name, azure_endpoint_url=azure_endpoint_url)
            if model_name == "text-embedding-3-large":
                self.out_dim = 3072
            if model_name == "ada-embeddings":
                self.out_dim = 1536
        elif model_name in ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2"]:
            self.model = sentence_transformers.SentenceTransformer(model_name_or_path=model_name, device=device)
            self.out_dim = self.model.get_sentence_embedding_dimension()
        else:
            assert False

    def forward(self, sentence=None):
        if self.model_name in ["text-embedding-3-large", "ada-embeddings"]:
            sentence_embed = self.model.generate_embedding(sentence)
        elif self.model_name in ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2"]:
            sentence_embed = self.model.encode(sentence, convert_to_numpy=False)

        return sentence_embed