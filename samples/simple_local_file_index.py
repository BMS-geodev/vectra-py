import os
import asyncio
from dataclasses import dataclass

from vectra_py import LocalDocumentIndex, LocalDocumentIndexConfig, CreateIndexConfig
from vectra_py.openai_embeddings import OpenAIEmbeddings, OpenAIEmbeddingsOptions
from vectra_py.gpt3_tokenizer import GPT3Tokenizer

# Embedding options
openai_options = OpenAIEmbeddingsOptions(
    api_key=os.environ.get("OPENAI_API_KEY"),
    model="text-embedding-ada-002",
    retry_policy=[2000, 5000],
    request_config={"timeout": 30}
)

# Paths to the files you want to add to the index
file_paths = [
    os.path.join(os.getcwd(), "hello.txt"),
    os.path.join(os.getcwd(), "hello1.txt"),
    os.path.join(os.getcwd(), "hello2.txt")
]

async def add_files_to_index(file_paths):
    """Add multiple files to the index without deleting the existing index."""

    # Initialize embeddings and tokenizer
    embeddings = OpenAIEmbeddings(options=openai_options)
    tokenizer = GPT3Tokenizer()

    # Setup document index configurations
    doc_index_config = LocalDocumentIndexConfig(
        folder_path=os.path.join(os.getcwd(), 'index'),
        tokenizer=tokenizer,
        embeddings=embeddings
    )
    simple_index_config = CreateIndexConfig(
        version=1,
        delete_if_exists=False,  # Keep existing index
        metadata_config={
            "model_framework": embeddings.__class__.__name__,
            "model_name": embeddings.options.model
        }
    )

    # Initialize and create index (only if it does not already exist)
    index = LocalDocumentIndex(doc_index_config)
    await index.init_index()
    
    try:
        await index.create_index(simple_index_config)  # Attempt to create the index if it doesn't exist
    except RuntimeError as e:
        print(f"Index already exists: {e}")

    # Add each file to the index
    for path in file_paths:
        try:
            with open(path, "r", encoding="utf-8") as file:
                text_content = file.read()
            
            # Use the filename as URI and add to index
            await index.upsert_document(
                uri=path,
                text=text_content,
                doc_type="text",
                metadata={
                    "filename": os.path.basename(path),
                    "description": "Sample text file"
                }
            )
            print(f"Document '{path}' added to index.")
        except FileNotFoundError:
            print(f"Error: '{path}' file not found.")
        except Exception as err:
            print(f"Error adding document '{path}': {err}")

async def main():
    await add_files_to_index(file_paths)

if __name__ == "__main__":
    asyncio.run(main())
