import os
import json
import asyncio
from typing import List
from dataclasses import dataclass

from vectra_py import LocalIndex, CreateIndexConfig, LocalDocumentIndex, LocalDocumentIndexConfig
from vectra_py.all_MiniLM_L6_v2_tokenizer import OSSTokenizer
from vectra_py.gpt3_tokenizer import GPT3Tokenizer
from vectra_py.oss_embeddings import OSSEmbeddings, OSSEmbeddingsOptions
from vectra_py.openai_embeddings import OpenAIEmbeddings, OpenAIEmbeddingsOptions
from vectra_py.file_fetcher import FileFetcher
from vectra_py.web_fetcher import WebFetcher

# Set paths for testing
keys_file = "vectra.keys"
uri = None
list_file = "test_filing_1.json"
item_type = "html"

# Embedding options
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


def get_item_list(uri: str, list_file: str, item_type: str) -> List[Filing]:
    """Retrieve a list of filings from a specified URI or list file."""
    if uri:
        return [Filing(url=uri, company_name="Unknown", form_type=item_type, filing_date="")]
    elif list_file:
        with open(list_file, "r", encoding="utf-8") as file:
            filings = json.load(file)['filings']
            return [Filing(**filing) for filing in filings]
    else:
        raise ValueError("Please provide a URI or a list file")


async def add_docs_to_index(uri: str = None, list_file: str = None, item_type: str = None):
    """Add documents to the index."""
    print("Adding Documents to Index")

    # Initialize embeddings and tokenizer
    embeddings = OpenAIEmbeddings(options=openai_options)
    tokenizer = None

    # Setup document index configurations
    doc_index_config = LocalDocumentIndexConfig(
        folder_path=os.path.join(os.getcwd(), 'index'),
        tokenizer=tokenizer,
        embeddings=embeddings
    )
    simple_index_config = CreateIndexConfig(
        version=1,
        delete_if_exists=True,
        metadata_config={
            "model_framework": embeddings.__class__.__name__,
            "model_name": embeddings.options.model
        }
    )
    index = LocalDocumentIndex(doc_index_config)
    await index.init_index()
    await index.create_index(simple_index_config)

    # Retrieve URIs
    filings = get_item_list(uri, list_file, item_type)
    print("URIs to fetch:", filings)

    # Fetch and add documents to the index
    file_fetcher = FileFetcher()
    web_fetcher = WebFetcher()
    for filing in filings:
        print('yeet')
        try:
            url = filing.url
            print(f"Fetching {url}")
            fetcher = web_fetcher if url.startswith("http") else file_fetcher
            fetched_doc = fetcher.fetch(url)
            await index.upsert_document(
                uri=url,
                text=fetched_doc,
                doc_type=item_type,
                metadata={
                    "company_name": filing.company_name,
                    "form_type": filing.form_type,
                    "filing_date": filing.filing_date
                }
            )
            print(f"Document {url} added to index.")
        except Exception as err:
            print(f"Error adding document: {url}\n{err}")


async def main():
    await add_docs_to_index(list_file=list_file, item_type=item_type)

if __name__ == "__main__":
    asyncio.run(main())
