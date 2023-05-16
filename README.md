# vectra-py
This is a faithful port of Steven Ickman's [Vectra](https://github.com/Stevenic/vectra) in memory vector index project. Only modifications were to port into python, adjust for format, and generate some python friendly example code. Below readme follows on from his, with similar pythonic adjustments.

Thanks for the inspiriation Steve!


# Vectra-py
Vectra-py is a local vector database for Python with features similar to [Pinecone](https://www.pinecone.io/) or [Qdrant](https://qdrant.tech/) but built using local files. Each Vectra index is a folder on disk. There's an `index.json` file in the folder that contains all the vectors for the index along with any indexed metadata.  When you create an index you can specify which metadata properties to index and only those fields will be stored in the `index.json` file. All of the other metadata for an item will be stored on disk in a separate file keyed by a GUID.

When queryng Vectra you'll be able to use the same subset of [Mongo DB query operators](https://www.mongodb.com/docs/manual/reference/operator/query/) that Pinecone supports and the results will be returned sorted by simularity. Every item in the index will first be filtered by metadata and then ranked for simularity. Even though every item is evaluated its all in memory so it should by nearly instantanious. Likely 1ms - 2ms for even a rather large index. Smaller indexes should be <1ms.

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
from src.local_index import LocalIndex

index = LocalIndex(os.path.join(os.getcwd(), 'index'))
```

Next, from inside an async function, create your index:

```python
if not index.isIndexCreated():
        index.createIndex()
```

Add some items to your index:

```python
import { OpenAIApi, Configuration } from 'openai';

const configuration = new Configuration({
    apiKey: `<YOUR_KEY>`,
});

const api = new OpenAIApi(configuration);

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
