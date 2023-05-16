"""
This file is the example entry point for the application.
"""
import os
import asyncio
import openai
import openai_async
import timeit
from dotenv import load_dotenv
from src.local_index import LocalIndex

# Start a simple timer to view execution time.
start = timeit.default_timer()

# use dotenv and an .env file or similar to store your API key
load_dotenv()
openai.api_key = os.environ.get("OPENAI_APIKEY")
print(openai.debug)

# declare your index here, if it exists.
index = LocalIndex(os.path.join(os.getcwd(), 'index'))

# change this to your own index payload
sample_index_payload = ['bob', 'orb', 'lob', 'cob']
# change this to your own query payload
sample_query_payload = ['zap', 'bob', 'rob']


def create_index():
    """
    Create the index if it doesn't exist.
    """
    if not index.isIndexCreated():
        index.createIndex()


async def get_vector(text: str):
    """
    Get the openai vector for a given text.
    Extract the embedding from the response.
    Return the embedding vector.
    """
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
    """
    Add an item to the index.
    """
    vector = await get_vector(text)
    metadata = {'text': text}
    print(vector, metadata)
    await index.insertItem({'vector': vector,
                            'metadata': metadata})


async def insert_payload():
    """
    Manage the insert payload.
    """
    for item in sample_index_payload:
        try:
            await add_item(item)
        except Exception as e:
            print('insert issue', e)


async def query(text: str):
    """
    Query the index for a given text.
    Print the result similarity score and associated text.
    """
    vector = await get_vector(text)
    results = await index.queryItems(vector, 3)
    if len(results) > 0:
        for result in results:
            print(f"[{result['score']}] \
                  {result.get('item')['metadata']['text']}")
    else:
        print("No results found.")


async def query_payload():
    """
    Manage the query payload.
    """
    for item in sample_query_payload:
        try:
            await query(item)
        except Exception as e:
            print('query issue', e)


async def main():
    create_index()
    await insert_payload()
    await query_payload()

if __name__ == '__main__':
    try:
        asyncio.run(main())

        stop = timeit.default_timer()
        execution_time = stop - start
        print(f"Program Executed in {str(execution_time)} seconds")
    except Exception as e:
        print('main issue', e)
