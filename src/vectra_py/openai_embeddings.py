import asyncio
import requests
from typing import List, Union, Dict


class BaseOpenAIEmbeddingsOptions:
    def __init__(self, retry_policy: List[int] = None, request_config: Dict = None):
        self.retry_policy = retry_policy if retry_policy else [2000, 5000]
        self.request_config = request_config if request_config else {}


class OpenAIEmbeddingsOptions(BaseOpenAIEmbeddingsOptions):
    def __init__(
        self,
        api_key: str,
        model: str,
        organization: str = None,
        endpoint: str = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.model = model
        self.organization = organization
        self.endpoint = endpoint


class AzureOpenAIEmbeddingsOptions(BaseOpenAIEmbeddingsOptions):
    def __init__(
        self,
        azure_api_key: str,
        azure_endpoint: str,
        azure_deployment: str,
        azure_api_version: str = "2023-05-15",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.azure_api_key = azure_api_key
        self.azure_endpoint = azure_endpoint
        self.azure_deployment = azure_deployment
        self.azure_api_version = azure_api_version


class EmbeddingsResponse:
    def __init__(self, status: str, output: List[float] = None, message: str = None):
        self.status = status
        self.output = output
        self.message = message


class CreateEmbeddingRequest:
    def __init__(self, input: Union[str, List[str]]):
        self.input = input


class CreateEmbeddingResponse:
    def __init__(self, data: List[Dict], model: str, usage: Dict):
        self.data = data
        self.model = model
        self.usage = usage


class OpenAIEmbeddings:
    def __init__(self, options: Union[OpenAIEmbeddingsOptions, AzureOpenAIEmbeddingsOptions]):
        self._use_azure = isinstance(options, AzureOpenAIEmbeddingsOptions)
        self.options = options
        self.user_agent = "AlphaWave"

    @property
    def max_tokens(self):
        return 8000

    async def create_embeddings(self, inputs: Union[str, List[str]]) -> EmbeddingsResponse:
        response = await self.create_embedding_request({"input": inputs})
        # convert the response.text to json
        json_response = response.json()
        data = response.json().get('data')
        if response.status_code < 300:
            return EmbeddingsResponse(
                status="success",
                output=[item["embedding"] for item in data],
                message={"model": json_response.get('model'),
                         "usage": json_response.get('usage')}
            )
        elif response.status_code == 429:
            return EmbeddingsResponse(
                status="rate_limited",
                output=None,
                message="The embeddings API returned a rate limit error.",
            )
        else:
            return EmbeddingsResponse(
                status="error",
                output=None,
                message=f"The embeddings API returned an error status of {response.status_code}: {response.statusText}",
            )

    async def create_embedding_request(self, request: CreateEmbeddingRequest):
        if self._use_azure:
            options = self.options
            url = f"{options.azure_endpoint}/openai/deployments/{options.azure_deployment}/embeddings?api-version={options.azure_api_version}"
            return self.post(url, request)
        else:
            options = self.options
            url = f"{options.endpoint or 'https://api.openai.com'}/v1/embeddings"
            request['model'] = options.model
            test = await self.post(url, request, retry_count=0)
            return test

    async def post(self, url: str, body: Dict, retry_count: int = 0):
        request_config = dict(self.options.request_config)

        request_headers = request_config.setdefault("headers", {})
        request_headers.setdefault("Content-Type", "application/json")
        request_headers.setdefault("User-Agent", self.user_agent)

        if self._use_azure:
            options = self.options
            request_headers["api-key"] = options.azure_api_key
        else:
            options = self.options
            request_headers["Authorization"] = f"Bearer {options.api_key}"
            if options.organization:
                request_headers["OpenAI-Organization"] = options.organization

        response = requests.post(url, json=body, **request_config)

        if response.status_code == 429 and isinstance(self.options.retry_policy, list) and retry_count < len(self.options.retry_policy):
            delay = self.options.retry_policy[retry_count]
            await asyncio.sleep(delay / 1000)
            return await self.post(url, body, retry_count + 1)
        else:
            return response
