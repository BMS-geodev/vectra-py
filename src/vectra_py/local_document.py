import asyncio
import os
import json


class LocalDocument:
    def __init__(self, folder_path, id, uri):
        self._folder_path = folder_path
        self._id = id
        self._uri = uri
        self._metadata = None
        self._text = None

    @property
    def folder_path(self):
        return self._folder_path

    @property
    def id(self):
        return self._id

    @property
    def uri(self):
        return self._uri

    async def has_metadata(self):
        try:
            await asyncio.to_thread(os.access, os.path.join(self.folder_path, f"{self.id}.json"), os.R_OK)
            return True
        except Exception as err:
            print(f'Error checking metadata for document "{self.uri}": {str(err)}')
            return False

    async def load_metadata(self):
        if self._metadata is None:
            try:
                with open(os.path.join(self.folder_path, f"{self.id}.json"), 'r') as file:
                    json_str = await asyncio.to_thread(file.read)
                    self._metadata = json.loads(json_str)
            except Exception as err:
                raise Exception(f'Error reading metadata for document "{self.uri}": {str(err)}')

        return self._metadata

    async def load_text(self):
        if self._text is None:
            try:
                with open(os.path.join(self.folder_path, f"{self.id}.txt"), 'r') as file:
                    self._text = await asyncio.to_thread(file.read)
            except Exception as err:
                raise Exception(f'Error reading text file for document "{self.uri}": {str(err)}')

        return self._text
