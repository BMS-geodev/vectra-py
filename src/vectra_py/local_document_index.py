import os
import pathlib
from pathlib import Path
import time
import aiofiles.os
import json
import asyncio
from uuid import uuid4
from gpt3_tokenizer import GPT3Tokenizer
from local_index import LocalIndex, CreateIndexConfig
from text_splitter import TextSplitter, TextSplitterConfig
from custom_types import (
    MetadataFilter,
    EmbeddingsModel,
    Tokenizer,
    MetadataTypes,
    EmbeddingsResponse,
    QueryResult,
    DocumentChunkMetadata,
    DocumentCatalogStats,
)
from local_document_result import LocalDocumentResult
from local_document import LocalDocument
from typing import Dict, Optional, List, Union
from dataclasses import dataclass


@dataclass
class DocumentQueryOptions:
    max_documents: Optional[int] = None
    max_chunks: Optional[int] = None
    filter: Optional[MetadataFilter] = None


@dataclass
class LocalDocumentIndexConfig:
    folder_path: str
    tokenizer: Tokenizer
    embeddings: Optional[EmbeddingsModel] = None
    chunking_config: Optional[TextSplitterConfig] = None


@dataclass
class DocumentCatalog:
    version: int
    count: int
    uri_to_id: Dict[str, str]
    id_to_uri: Dict[str, str]


def is_catalog_created():
    # TODO: pass in appropriate path
    catalog_path = "/Users/brian/Documents/GitHub/vectra-py/index/catalog.json"
    exists = os.path.exists(catalog_path)
    if exists:
        print(f"exists: {exists}")
    # time.sleep(1)
    return exists


class LocalDocumentIndex(LocalIndex):
    def __init__(self, doc_index_config: LocalDocumentIndexConfig):
        super().__init__(doc_index_config.folder_path)
        self._embeddings = doc_index_config.embeddings
        self._chunking_config = {
            "keep_separators": True,
            "chunk_size": 512,
            "chunk_overlap": 0,
            **(doc_index_config.chunking_config or {}),
        }
        self._tokenizer = doc_index_config.tokenizer or self._chunking_config.get("tokenizer") or GPT3Tokenizer()
        self._chunking_config["tokenizer"] = self._tokenizer
        self._catalog = None
        self._new_catalog = None

    async def get_document_id(self, uri: str) -> Optional[str]:
        await self.load_index_data()
        return self._catalog["uri_to_id"].get(uri)

    async def get_document_uri(self, document_id: str) -> Optional[str]:
        await self.load_index_data()
        return self._catalog.id_to_uri.get(document_id)

    async def create_index(self, config: Optional[CreateIndexConfig] = None) -> None:
        await super().create_index(config)
        await self.load_index_data()

    async def delete_document(self, uri: str) -> None:
        document_id = await self.get_document_id(uri)
        if document_id is None:
            return

        await self.begin_update()
        try:
            chunks = await self.list_items_by_metadata(DocumentChunkMetadata(document_id=document_id))
            for chunk in chunks:
                await self.deleteItem(chunk.id)

            del self._new_catalog.uri_to_id[uri]
            del self._new_catalog.id_to_uri[document_id]
            self._new_catalog.count -= 1

            await self.end_update()
        except Exception as err:
            self.cancel_update()
            raise Exception(f'Error deleting document "{uri}": {str(err)}')

        try:
            os.unlink(os.path.join(self.folder_path, f'{document_id}.txt'))
        except Exception as err:
            raise Exception(f'Error removing text file for document "{uri}" from disk: {str(err)}')

        try:
            os.unlink(os.path.join(self.folder_path, f'{document_id}.json'))
        except Exception as err:
            raise Exception(f'Error removing json metadata file for document "{uri}" from disk: {str(err)}')

    async def get_catalog_stats(self) -> DocumentCatalogStats:
        stats = await self.getIndexStats()
        return DocumentCatalogStats(
            version=self._catalog.version,
            documents=self._catalog.count,
            chunks=stats.items,
            metadata_config=stats.metadata_config,
        )

    async def upsert_document(
        self,
        uri: str,
        text: str,
        doc_type: Optional[str] = None,
        metadata: Optional[Dict[str, MetadataTypes]] = None
    ) -> LocalDocument:
        if not self._embeddings:
            raise Exception('Embeddings model not configured.')

        document_id = await self.get_document_id(uri)
        if document_id is not None:
            await self.delete_document(uri)
        else:
            document_id = str(uuid4())

        config = {
            **(self._chunking_config or {}),
            "doc_type": doc_type or self._chunking_config.get("doc_type"),
        }

        if config["doc_type"] is None:
            pos = uri.rfind('.')
            if pos >= 0:
                ext = uri[pos + 1:].lower()
                config["doc_type"] = ext

        splitter = TextSplitter(config)
        chunks = splitter.split(text)
        total_tokens = 0
        chunk_batches = []
        current_batch = []

        for chunk in chunks:
            total_tokens += len(chunk.tokens)

            if total_tokens > self._embeddings.max_tokens:
                chunk_batches.append(current_batch)
                current_batch = []
                total_tokens = len(chunk.tokens)

            current_batch.append(chunk.text.replace('\n', ' '))

        if current_batch:
            chunk_batches.append(current_batch)

        embeddings = []

        for batch in chunk_batches:
            try:
                response = await self._embeddings.create_embeddings(batch)
            except Exception as err:
                raise Exception(f'Error generating embeddings: {str(err)}')

            if response.status != 'success':
                raise Exception(f'Error generating embeddings: {response.message}')

            embeddings.extend(response.output or [])

        await self.begin_update()
        try:
            for i, chunk in enumerate(chunks):
                embedding = embeddings[i]
                chunk_metadata = {
                    "document_id": document_id,
                    "start_pos": chunk.start_pos,
                    "end_pos": chunk.end_pos,
                    **(metadata or {}),
                }
                await self.insert_item(
                    {
                        "id": str(uuid4()),
                        "metadata": chunk_metadata,
                        "vector": embedding,
                    }
                )
            if metadata:
                with open(os.path.join(self.folder_path, f'{document_id}.json'), 'w') as metadata_file:
                    json.dump(metadata, metadata_file)

            with open(os.path.join(self.folder_path, f'{document_id}.txt'), 'w') as text_file:
                text_file.write(text)

            self._new_catalog['uri_to_id'][uri] = document_id
            self._new_catalog['id_to_uri'][document_id] = uri
            self._new_catalog['count'] += 1

            await self.end_update()
        except Exception as err:
            self.cancel_update()
            raise Exception(f'Error adding document "{uri}": {str(err)}')

        return LocalDocument(self.folder_path, document_id, uri)

    async def query_documents(self, query: str, options: DocumentQueryOptions = None) -> List[LocalDocumentResult]:
        if not self._embeddings:
            raise Exception('Embeddings model not configured.')

        options = options or DocumentQueryOptions(max_documents=10, max_chunks=50)

        try:
            embeddings = await self._embeddings.create_embeddings(query.replace('\n', ' '))
        except Exception as err:
            raise Exception(f'Error generating embeddings for query: {str(err)}')

        if embeddings.status != 'success':
            raise Exception(f'Error generating embeddings for query: {embeddings.message}')

        results = await self.query_items(embeddings.output[0], options.max_chunks, options.filter)
        document_chunks = {}

        for result in results:
            metadata = result.item.metadata

            if metadata.document_id not in document_chunks:
                document_chunks[metadata.document_id] = []

            document_chunks[metadata.document_id].append(result)

        document_results = []

        for document_id, chunks in document_chunks.items():
            uri = await self.get_document_uri(document_id)
            document_result = LocalDocumentResult(self.folder_path, document_id, uri, chunks, self._tokenizer)
            document_results.append(document_result)

        document_results.sort(key=lambda x: x.score, reverse=True)
        return document_results[:options.max_documents]

    async def begin_update(self):
        await super().begin_update()
        self._new_catalog = self._catalog.copy()

    def cancel_update(self):
        super().cancel_update()
        self._new_catalog = None

    async def end_update(self):
        await super().end_update()

        try:
            # Save catalog
            catalog_path = os.path.join(self.folder_path, 'catalog.json')
            with open(catalog_path, 'w') as catalog_file:
                json.dump(self._new_catalog, catalog_file)
            self._catalog = self._new_catalog
            self._new_catalog = None
        except Exception as err:
            raise Exception(f'Error saving document catalog: {str(err)}')

    async def load_index_data(self):
        await super().load_index_data()

        if self._catalog:
            return

        catalog_path = os.path.join(self.folder_path, 'catalog.json')
        thread_test = await asyncio.gather(
                                            asyncio.to_thread(is_catalog_created),
                                            asyncio.sleep(1)
                                            )
        if is_catalog_created():
            # Load catalog
            async with aiofiles.open(catalog_path, 'r') as catalog_file:
                contents = await catalog_file.read()
                self._catalog = json.loads(contents)
        else:
            try:
                # Initialize catalog
                self._catalog = {
                    'version': 1,
                    'count': 0,
                    'uri_to_id': {},
                    'id_to_uri': {},
                }
                with open(catalog_path, 'w') as catalog_file:
                    json.dump(self._catalog, catalog_file)
            except Exception as err:
                raise Exception(f'Error creating document catalog: {str(err)}')
