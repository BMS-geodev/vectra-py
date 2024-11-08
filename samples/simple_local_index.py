import os
import asyncio
from vectra_py import LocalIndex, CreateIndexConfig
import openai_async
import openai
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# Initialize OpenAI API key
openai.api_key = os.getenv("OPENAI_APIKEY")

# Initialize LocalIndex
index_path = os.path.join(os.getcwd(), 'index')
index = LocalIndex(index_path)
print(index.__dict__)

# Function to retrieve vector embedding from OpenAI API
async def get_vector(text: str):
    print(f"Getting vector for: {text}")
    model = "text-embedding-ada-002"
    response = await openai_async.embeddings(
        openai.api_key,
        timeout=2,
        payload={"model": model, "input": [text]},
    )
    return response.json()['data'][0]['embedding']

# Function to add an item to the index
async def add_item(text: str):
    vector = await get_vector(text)
    metadata = {'text': text}
    truncated_vector = vector[:10]
    print(f"Adding item with vector: {truncated_vector + ['...']} and metadata: {metadata}")
    await index.insert_item({'vector': vector, 'metadata': metadata})

# Function to query similar items from the index
async def query(text: str):
    vector = await get_vector(text)
    results = await index.query_items(vector, top_k=3)
    if results:
        for result in results:
            print(f"[{result['score']}] {result['item']['metadata']['text']}")
    else:
        print("No results found.")

async def main():
    await index.init_index()

    # Add items to the index
    await add_item('apple')
    await add_item('oranges')
    await add_item('red')
    await add_item('blue')

    # Query the index
    print("Querying for 'green':")
    await query('green')
    
    print("\nQuerying for 'banana':")
    await query('banana')

# Run the main function in an async context
asyncio.run(main())
