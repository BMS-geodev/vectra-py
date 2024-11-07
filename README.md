Update: Revisiting this very stale project in Fall 2024. Going to start with a general review of the source vectra project, then go from there. As such I'll probably reply to the issues and close them, pending new discussion. 

# vectra-py
This is a faithful port of Steven Ickman's [Vectra](https://github.com/Stevenic/vectra) in memory vector index project. Only modifications were to port into python, adjust for format, and generate some python friendly example code. Below readme follows on from his, with similar pythonic adjustments.

Thanks for the inspiriation Steve!


Vectra-py is a local vector database for Python with features similar to [Pinecone](https://www.pinecone.io/) or [Qdrant](https://qdrant.tech/) but built using local files. Each Vectra index is a folder on disk. There's an `index.json` file in the folder that contains all the vectors for the index along with any indexed metadata.  When you create an index you can specify which metadata properties to index and only those fields will be stored in the `index.json` file. All of the other metadata for an item will be stored on disk in a separate file keyed by a GUID.

When queryng Vectra you'll be able to use the same subset of [Mongo DB query operators](https://www.mongodb.com/docs/manual/reference/operator/query/) that Pinecone supports and the results will be returned sorted by similarity. Every item in the index will first be filtered by metadata and then ranked for similarity. Even though every item is evaluated its all in memory so it should by nearly instantanious. Likely 1ms - 2ms for even a rather large index. Smaller indexes should be <1ms.

Keep in mind that your entire Vectra index is loaded into memory so it's not well suited for scenarios like long term chat bot memory. Use a real vector DB for that. Vectra is intended to be used in scenarios where you have a small corpus of mostly static data that you'd like to include in your prompt. Infinite few shot examples would be a great use case for Vectra or even just a single document you want to ask questions over.

Pinecone style namespaces aren't directly supported but you could easily mimic them by creating a separate Vectra index (and folder) for each namespace.

## Installation

```
$ pip install vectra-py
```

## Prep

Use dotenv or set env var to store your openAI API Key.

## Usage

First create an instance of `LocalIndex` with the path to the folder where you want you're items stored:

```python
from vectra_py import LocalIndex

index = LocalIndex(os.path.join(os.getcwd(), 'index'))
```

Next, from inside an async function, create your index:

```python
if not index.isIndexCreated():
        index.createIndex()
```

Add some items to your index:

```python
openai.api_key = os.environ.get("OPENAI_APIKEY")

async def get_vector(text: str):
    print(text)
    model = "text-embedding-ada-002"
    response = await openai_async.embeddings(
                                            openai.api_key,
                                            timeout=2,
                                            payload={"model": model,
                                                     "input": [text]},
                                        )
    return response.json()['data'][0]['embedding']


async def add_item(text: str):
    vector = await get_vector(text)
    metadata = {'text': text}
    print(vector, metadata)
    await index.insertItem({'vector': vector,
                            'metadata': metadata})

// Add items
await add_item('apple');
await add_item('oranges');
await add_item('red');
await add_item('blue');
```

Then query for items:

```python
async def query(text: str):
    vector = await get_vector(text)
    results = await index.queryItems(vector, 3)
    if len(results) > 0:
        for result in results:
            print(f"[{result['score']}] \
                  {result.get('item')['metadata']['text']}")
    else:
        print("No results found.")

await query('green')
/*
[0.9036569942401076] blue
[0.8758153664568566] red
[0.8323828606103998] apple
*/

await query('banana')
/*
[0.9033128691220631] apple
[0.8493374123092652] oranges
[0.8415324469533297] blue
*/
```

Creating a document index is a bit more involved. 

First, set up configurations. Pass in an example list of Filing objects as a list_file like:
```json
{
    "filings": [
        {
            "company_name": "DigitalBridge Group, Inc.",
            "form_type": "10-Q",
            "filing_date": "20230505",
            "url": "https://www.sec.gov/Archives/edgar/data/0001679688/000167968823000049/dbrg-20230331.htm"
        }
    ]
}
```

```python
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
```

Next, write a basic way to organize the filings.
```python
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
```

Then, handle the operations to create, manage, and populate the doc index.

```python
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

```
