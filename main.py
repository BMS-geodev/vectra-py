"""
This file is the example entry point for the application. 
"""
import os
import asyncio
import openai
import openai_async
import timeit

start = timeit.default_timer()

from src.local_index import LocalIndex

openai.api_key = "***REMOVED***"

print(os.path.join(os.getcwd(), 'index'))
index = LocalIndex(os.path.join(os.getcwd(), 'index'))
print(index)

def create_index():
    if not index.isIndexCreated():
        index.createIndex()
    # print(index.isIndexCreated())

async def get_vector(text: str):
    response = await openai_async.embeddings(
                                            openai.api_key,
                                            timeout=2,
                                            payload={"model": "text-embedding-ada-002", 
                                                     "input": [text]},
                                        )
    return response.json()['data'][0]['embedding']

async def add_item(text: str):
    vector = await get_vector(text)
    metadata = {'text': text}
    await index.insertItem({'vector': vector, 
                            'metadata': metadata})

async def insert_payload():
    try:
        await add_item('apple')
        await add_item('oranges')
        await add_item('red')
        await add_item('blue')
    except Exception as e:
        print('insert issue', e)

async def query(text: str):
    vector = await get_vector(text)
    results = await index.queryItems(vector, 3)
    if len(results) > 0:
        for result in results:
            print(f"[{result['score']}] {result.get('item')['metadata']['text']}")
    else:
        print("No results found.")

async def main():
    create_index()
    await insert_payload()
    await query('green')
    await query('banana')

if __name__ == '__main__':
    try:
        asyncio.run(main())
        stop = timeit.default_timer()
        execution_time = stop - start

        print("Program Executed in "+str(execution_time)) # It returns time in seconds
    except Exception as e:
        print('main issue', e)