import asyncio
import requests
from typing import List, Union, Dict
from .all_MiniLM_L6_v2_tokenizer import OSSTokenizer


class BaseOSSEmbeddingsOptions:
    def __init__(self, retry_policy: List[int] = None, request_config: Dict = None):
        self.retry_policy = retry_policy if retry_policy else [2000, 5000]
        self.request_config = request_config if request_config else {}


class OSSEmbeddingsOptions(BaseOSSEmbeddingsOptions):
    def __init__(
        self,
        model: str,
        tokenizer: OSSTokenizer,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tokenizer = OSSTokenizer(model_name=model)
        self.model = model


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


class OSSEmbeddings:
    def __init__(self, options: OSSEmbeddingsOptions):
        self._local = True  # use a locally stored model
        self.options = options
        self.model = options.model
        self.tokenizer = options.tokenizer
        # self.user_agent = "AlphaWave"

    @property
    def max_tokens(self):
        return 8000
    
    async def create_embeddings(self, inputs: Union[str, List[str]]) -> EmbeddingsResponse:
        # create embeddings from the local model
        try:
            data = [self.options.tokenizer.encode(item) for item in inputs]
            return EmbeddingsResponse(
                status="success",
                output=data,
                message={"model": self.model,
                         "usage": 'unknown'}
            )
        except Exception as e:
            print('OSS encoding error', e)
            return EmbeddingsResponse(
                status="error",
                output=None,
                message=f"Encoding error: {e}",
            )


    # async def create_embeddings(self, inputs: Union[str, List[str]]) -> EmbeddingsResponse:
    #     # print('openai create_embeddings', inputs)
    #     print('create_embeddings', OSEmbeddingsOptions.__dict__)
    #     response = await self.create_embedding_request({"input": inputs})
    #     # convert the response.text to json
    #     json_response = response.json()
    #     data = response.json().get('data')
    #     if response.status_code < 300:
    #         return EmbeddingsResponse(
    #             status="success",
    #             output=[item["embedding"] for item in data],
    #             message={"model": json_response.get('model'),
    #                      "usage": json_response.get('usage')}
    #         )
    #     elif response.status_code == 429:
    #         return EmbeddingsResponse(
    #             status="rate_limited",
    #             output=None,
    #             message="The embeddings API returned a rate limit error.",
    #         )
    #     else:
    #         return EmbeddingsResponse(
    #             status="error",
    #             output=None,
    #             message=f"The embeddings API returned an error status of {response.status_code}: {response.statusText}",
    #         )

    # async def create_embedding_request(self, request: CreateEmbeddingRequest):
    #     # print('openai create_embedding_request', request)
    #     if self._use_azure:
    #         options = self.options
    #         url = f"{options.azure_endpoint}/openai/deployments/{options.azure_deployment}/embeddings?api-version={options.azure_api_version}"
    #         return self.post(url, request)
    #     else:
    #         # print('else', self.options.__dict__)
    #         options = self.options
    #         # print('openai create_embedding_request else', options.__dict__)
    #         url = f"{options.endpoint or 'https://api.openai.com'}/v1/embeddings"
    #         # print('---------------openai create_embedding_request else request', request.keys())
    #         # print('---------------openai create_embedding_request else, options', options.model)
    #         request['model'] = options.model
    #         # print('zaza')
    #         # print(options.model)
    #         test = await self.post(url, request, retry_count=0)
    #         return test

    # async def post(self, url: str, body: Dict, retry_count: int = 0):
    #     # print('openai post', url, body, retry_count)
    #     request_config = dict(self.options.request_config)

    #     request_headers = request_config.setdefault("headers", {})
    #     request_headers.setdefault("Content-Type", "application/json")
    #     request_headers.setdefault("User-Agent", self.user_agent)

    #     if self._use_azure:
    #         options = self.options
    #         request_headers["api-key"] = options.azure_api_key
    #     else:
    #         options = self.options
    #         request_headers["Authorization"] = f"Bearer {options.api_key}"
    #         if options.organization:
    #             request_headers["OpenAI-Organization"] = options.organization

    #     response = requests.post(url, json=body, **request_config)
    #     # print('post', response.__dict__.keys())

    #     if response.status_code == 429 and isinstance(self.options.retry_policy, list) and retry_count < len(self.options.retry_policy):
    #         delay = self.options.retry_policy[retry_count]
    #         await asyncio.sleep(delay / 1000)
    #         return await self.post(url, body, retry_count + 1)
    #     else:
    #         return response
