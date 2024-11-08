import os
import asyncio
from vectra_py import LocalDocumentIndex, LocalDocumentResult, LocalDocumentIndexConfig, DocumentQueryOptions
from vectra_py import OSSEmbeddings, OSSEmbeddingsOptions, OSSTokenizer  # Assuming you're using OSS embeddings
from vectra_py.gpt3_tokenizer import GPT3Tokenizer
from vectra_py.openai_embeddings import OpenAIEmbeddings, OpenAIEmbeddingsOptions

# Define the configuration for the embeddings and tokenizer
oss_options = OSSEmbeddingsOptions(
    tokenizer=OSSTokenizer(model_name="sentence-transformers/all-MiniLM-L6-v2"),
    model="sentence-transformers/all-MiniLM-L6-v2"
)

# Embedding options
openai_options = OpenAIEmbeddingsOptions(
    api_key=os.environ.get("OPENAI_API_KEY"),
    model="text-embedding-ada-002",
    retry_policy=[2000, 5000],
    request_config={"timeout": 30}
)

# Initialize embeddings and tokenizer
embeddings = OpenAIEmbeddings(options=openai_options)
tokenizer = None

# Path to the index folder (where your index is stored)
index_folder_path = "./index"  # Update with your actual path

# Initialize the document index configuration
doc_index_config = LocalDocumentIndexConfig(
    folder_path=index_folder_path,
    embeddings=embeddings,
    tokenizer=None
)

# Initialize the document index
index = LocalDocumentIndex(doc_index_config)

# Define a query function to search the index and render results
async def query_index(query_text: str, max_tokens: int = 100, max_sections: int = 3):
    # Define the query options
    query_options = DocumentQueryOptions(
        max_documents=5,     # Retrieve the top 5 documents
        max_chunks=50        # Limit the number of chunks per document
    )

    # Perform the query
    results = await index.query_documents(query_text, query_options)

    # Display results
    for result in results:
        print(f"Document URI: {result.uri}")
        print(f"Average Score: {result.score:.2f}")

        # Render the top sections of the document
        document_result = LocalDocumentResult(
            index=index,
            id=result.id,
            uri=result.uri,
            chunks=result.chunks,
            tokenizer=GPT3Tokenizer()  # Use GPT-3 tokenizer for rendering
        )
        sections = await document_result.render_sections(max_tokens, max_sections)
        
        # Print the sections
        for i, section in enumerate(sections, 1):
            print(f"\nSection {i} (Score: {section['score']:.2f})")
            print(section['text'])
            print("-" * 50)

# Run the query with an example search term
if __name__ == "__main__":
    search_query = "b"  # Replace with your actual query
    asyncio.run(query_index(search_query))
