import os


class FileFetcher:
    async def fetch(self, uri):
        # Check if the path exists and whether it's a directory
        if os.path.exists(uri):
            if os.path.isdir(uri):
                # If it's a directory, read all files and recurse
                files = os.listdir(uri)
                for file in files:
                    file_path = os.path.join(uri, file)
                    await self.fetch(file_path)
                return True
            else:
                # If it's a file, read its contents
                with open(uri, 'r', encoding='utf-8') as file:
                    text = file.read()
                    # Determine the document type based on the file extension
                    _, file_extension = os.path.splitext(uri)
                    doc_type = file_extension[1:].lower() if file_extension else None
                    return uri, text, doc_type
        else:
            # Handle the case where the path doesn't exist
            return None
