import argparse
import json
import os

from local_document_index import LocalDocumentIndex
from web_fetcher import WebFetcher
from openai_embeddings import OpenAIEmbeddings
from file_fetcher import FileFetcher


async def run():
    parser = argparse.ArgumentParser(prog="vectra")
    subparsers = parser.add_subparsers(dest="command")

    # Create command
    create_parser = subparsers.add_parser("create", description="Create a new local index")
    create_parser.add_argument("index", type=str, help="Path to the index folder")

    # Delete command
    delete_parser = subparsers.add_parser("delete", description="Delete an existing local index")
    delete_parser.add_argument("index", type=str, help="Path to the index folder")

    # Add command
    add_parser = subparsers.add_parser("add", description="Add one or more web pages to an index")
    add_parser.add_argument("index", type=str, help="Path to the index folder")
    add_parser.add_argument("--keys", "-k", type=str, required=True, help="Path to a JSON file containing model keys")
    add_parser.add_argument("--uri", "-u", type=str, nargs="+", help="HTTP/HTTPS links to web pages")
    add_parser.add_argument("--list", "-l", type=str, help="Path to a file containing a list of web pages")
    add_parser.add_argument("--chunk-size", "-cs", type=int, default=512, help="Size of generated chunks in tokens")

    # Remove command
    remove_parser = subparsers.add_parser("remove", description="Remove one or more documents from an index")
    remove_parser.add_argument("index", type=str, help="Path to the index folder")
    remove_parser.add_argument("--uri", "-u", type=str, nargs="+", help="URIs of documents to remove")
    remove_parser.add_argument("--list", "-l", type=str, help="Path to a file containing a list of documents to remove")

    # Stats command
    stats_parser = subparsers.add_parser("stats", description="Print the stats for a local index")
    stats_parser.add_argument("index", type=str, help="Path to the index folder")

    # Query command
    query_parser = subparsers.add_parser("query", description="Query a local index")
    query_parser.add_argument("index", type=str, help="Path to the index folder")
    query_parser.add_argument("query", type=str, help="Query text")
    query_parser.add_argument("--keys", "-k", type=str, required=True, help="Path to a JSON file containing model keys")
    query_parser.add_argument("--document-count", "-dc", type=int, default=10, help="Max number of documents to return")
    query_parser.add_argument("--chunk-count", "-cc", type=int, default=50, help="Max number of chunks to return")
    query_parser.add_argument("--section-count", "-sc", type=int, default=1, help="Max number of document sections to render")
    query_parser.add_argument("--tokens", "-t", type=int, default=2000, help="Max number of tokens to render for each section")
    query_parser.add_argument("--format", "-f", type=str, default="sections", choices=["sections", "stats", "chunks"], help="Format of the rendered results")

    args = parser.parse_args()

    if args.command == "create":
        folder_path = args.index
        index = LocalDocumentIndex(folder_path)
        print(f"Creating index at {folder_path}")
        await index.create_index(version=1, delete_if_exists=True)

    elif args.command == "delete":
        folder_path = args.index
        print(f"Deleting index at {folder_path}")
        index = LocalDocumentIndex(folder_path)
        await index.delete_index()

    elif args.command == "add":
        print("Adding Web Pages to Index")

        # Create embeddings
        with open(args.keys, "r", encoding="utf-8") as keys_file:
            keys = json.load(keys_file)
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", **keys)

        # Initialize index
        folder_path = args.index
        index = LocalDocumentIndex(
            folder_path=folder_path,
            embeddings=embeddings,
            chunking_config={"chunk_size": args.chunk_size},
        )

        # Get list of URIs
        uris = get_item_list(args.uri, args.list, "web page")

        # Fetch web pages
        file_fetcher = FileFetcher()
        web_fetcher = WebFetcher()
        for uri in uris:
            try:
                print(f"Fetching {uri}")
                fetcher = web_fetcher if uri.startswith("http") else file_fetcher
                await fetcher.fetch(uri, index_upsert_document(index))
            except Exception as err:
                print(f"Error adding: {uri}\n{str(err)}")

    elif args.command == "remove":
        folder_path = args.index
        index = LocalDocumentIndex(folder_path)

        # Get list of URIs
        uris = get_item_list(args.uri, args.list, "document")

        # Remove documents
        for uri in uris:
            print(f"Removing {uri}")
            await index.delete_document(uri)

    elif args.command == "stats":
        folder_path = args.index
        index = LocalDocumentIndex(folder_path)
        stats = await index.get_catalog_stats()
        print("Index Stats")
        print(stats)

    elif args.command == "query":
        print("Querying Index")

        # Create embeddings
        with open(args.keys, "r", encoding="utf-8") as keys_file:
            keys = json.load(keys_file)
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", **keys)

        # Initialize index
        folder_path = args.index
        index = LocalDocumentIndex(folder_path=folder_path, embeddings=embeddings)

        # Query index
        query = args.query
        results = await index.query_documents(
            query,
            max_documents=args.document_count,
            max_chunks=args.chunk_count,
        )

        # Render results
        for result in results:
            print(result.uri)
            print("score:", result.score)
            print("chunks:", len(result.chunks))
            if args.format == "sections":
                sections = await result.render_sections(args.tokens, args.section_count)
                for i, section in enumerate(sections):
                    print(f"Section {i + 1}" if args.section_count > 1 else "Section")
                    print("score:", section.score)
                    print("tokens:", section.token_count)
                    print(section.text)
            elif args.format == "chunks":
                text = await result.load_text()
                for i, chunk in enumerate(result.chunks):
                    start_pos = chunk.item.metadata["startPos"]
                    end_pos = chunk.item.metadata["endPos"]
                    print(f"Chunk {i + 1}")
                    print("score:", chunk.score)
                    print("startPos:", start_pos)
                    print("endPos:", end_pos)
                    print(text[start_pos:end_pos + 1])


def get_item_list(items, list_file, item_type):
    if items is not None and len(items) > 0:
        return items
    elif list_file is not None and list_file.strip():
        with open(list_file, "r", encoding="utf-8") as file:
            item_list = [line.strip() for line in file.readlines() if line.strip()]
        return item_list
    else:
        raise ValueError(f"You must specify either one or more '--uri <{item_type}>' for the items or a '--list <file path>' for a file containing the items.")


def index_upsert_document(index):
    async def upsert_document(uri, text, doc_type):
        print(f"Indexing {uri}")
        await index.upsert_document(uri, text, doc_type)
        print(f"Added {uri}")
        return True

    return upsert_document

if __name__ == "__main__":
    run()
