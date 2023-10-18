from dataclasses import dataclass
from typing import List, Union, Dict, Optional, Any


@dataclass
class EmbeddingsModel:
    max_tokens: int

    # async def create_embeddings(self, inputs: Union[str, List[str]]) -> 'EmbeddingsResponse':
    #     pass


@dataclass
class EmbeddingsResponse:
    status: str
    output: List[List[float]] = None
    message: str = None


@dataclass
class TextChunk:
    text: str
    tokens: List[int]
    start_pos: int
    end_pos: int
    start_overlap: List[int]
    end_overlap: List[int]


@dataclass
class TextFetcher:
    async def fetch(self, uri: str) -> Dict[str, Union[str, None]]:
        pass


@dataclass
class IndexStats:
    version: int
    metadata_config: Dict[str, Optional[List[str]]]
    items: int


@dataclass
class IndexItem:
    id: str
    metadata: Dict[str, Any]
    vector: List[float]
    norm: float
    metadata_file: str = None


@dataclass
class MetadataFilter:
    eq: Union[int, str, bool] = None  # Equal to (number, string, boolean)
    ne: Union[int, str, bool] = None  # Not equal to (number, string, boolean)
    gt: int = None  # Greater than (number)
    gte: int = None  # Greater than or equal to (number)
    lt: int = None  # Less than (number)
    lte: int = None  # Less than or equal to (number)
    _in: List[Union[int, str]] = None  # In array (string or number)
    nin: List[Union[int, str]] = None  # Not in array (string or number)
    _and: List['MetadataFilter'] = None  # AND (MetadataFilter[])
    _or: List['MetadataFilter'] = None  # OR (MetadataFilter[])
    extra: Dict[str, Any] = None


@dataclass
class MetadataTypes:
    value: Union[int, str, bool]


@dataclass
class QueryResult:
    item: IndexItem
    score: float


@dataclass
class Tokenizer:
    def decode(self, tokens: List[int]) -> str:
        pass

    def encode(self, text: str) -> List[int]:
        pass


@dataclass
class DocumentChunkMetadata:
    document_id: str
    start_pos: int
    end_pos: int
    extra: Dict[str, Any] = None


@dataclass
class DocumentCatalogStats:
    version: int
    documents: int
    chunks: int
    metadata_config: Dict[str, Optional[List[str]]]
    extra: Dict[str, Any] = None


@dataclass
class DocumentTextSection:
    text: str
    token_count: int
    score: float
