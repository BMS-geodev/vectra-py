# Truncated operational pipeline to load docs to a local index.
# Created to reduce variables and test the vectra-py operations.
# Vectra-cli.py is the eventual operational entrypoint.

import os
import json
import asyncio
from typing import List
from dataclasses import dataclass

from all_MiniLM_L6_v2_tokenizer import OSSTokenizer
from oss_embeddings import OSSEmbeddings, OSSEmbeddingsOptions
from openai_embeddings import OpenAIEmbeddings, OpenAIEmbeddingsOptions
from local_index import LocalIndex, CreateIndexConfig
from local_document_index import LocalDocumentIndex, LocalDocumentIndexConfig
# from local_index import CreateIndexConfig
from file_fetcher import FileFetcher
from web_fetcher import WebFetcher

# test defaults
keys_file = "vectra.keys"
uri = None
list_file = "test_filings_1.json"
item_type = "html"

openai_options = OpenAIEmbeddingsOptions(
    api_key=os.environ.get("OPENAI_API_KEY"),
    model="text-embedding-ada-002",
    retry_policy=[2000, 5000],
    request_config={"timeout": 30}
)

oss_options = OSSEmbeddingsOptions(
    tokenizer=OSSTokenizer(model_name="sentence-transformers/all-MiniLM-L6-v2"),
    model="sentence-transformers/all-MiniLM-L6-v2"
)


@dataclass
class Filing:
    company_name: str
    form_type: str
    filing_date: str
    url: str


def get_item_list(uri: str, list_file: str, item_type: str) -> List[str]:
    """Get a list of URIs from a specified URI or list file"""
    if uri:
        return [uri]
    elif list_file:
        with open(list_file, "r", encoding="utf-8") as file:
            filings = json.load(file)['filings']
            return [Filing(**filing) for filing in filings]

    else:
        raise Exception(f"Please provide a {item_type} URI or list file")


async def add_docs_to_index(uri: str = None, list_file: str = None, item_type: str = None):
    """
    Handle operations.
    Establish the index, prepare the config, fetch the docs, and add them to the index.
    """
    print("Adding Web Pages to Index")

    # Create embeddings and tokenizer
    # embeddings = OpenAIEmbeddings(options=openai_options)
    # tokenizer = None  # the tokenizer is wrapped in the openai embedding.
    embeddings = OSSEmbeddings(options=oss_options)
    tokenizer = embeddings.tokenizer
    # Initialize index in current directory
    # update the index_config to include the embeddings
    doc_index_config = LocalDocumentIndexConfig(folder_path=(os.path.join(os.getcwd(), 'index')),
                                                tokenizer=tokenizer,
                                                embeddings=embeddings)
    simple_index_config = CreateIndexConfig(version=1, 
                                            delete_if_exists=True,
                                            metadata_config={"model_framework": embeddings.__class__.__name__,
                                                             "model_name": embeddings.options.model},
                                            )
    index = LocalDocumentIndex(doc_index_config)
    await index.create_index(simple_index_config)

    # Get list of URIs
    uris = get_item_list(uri, list_file, item_type)
    print('uris', uris)

    # Fetch web pages
    file_fetcher = FileFetcher()
    web_fetcher = WebFetcher()
    for uri in uris:
        try:
            url = uri.url if isinstance(uri, Filing) else uri
            print(f"Fetching {url}")
            fetcher = web_fetcher if url.startswith("http") else file_fetcher
            fetched_doc = fetcher.fetch(url)
            await index.upsert_document(url,
                                        fetched_doc,
                                        doc_type=item_type)
        except Exception as err:
            print(f"Error adding: {uri}\n{str(err)}")


async def main():
    await add_docs_to_index(list_file=list_file, item_type=item_type)

if __name__ == "__main__":
    asyncio.run(main())
