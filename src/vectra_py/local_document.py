import asyncio
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
from .custom_types import MetadataTypes


class LocalDocument:
    def __init__(self, index, id: str, uri: str):
        self._index = index
        self._id = id
        self._uri = uri
        self._metadata: Optional[Dict[str, MetadataTypes]] = None
        self._text: Optional[str] = None

    @property
    def folder_path(self) -> str:
        return self._index.folder_path

    @property
    def id(self) -> str:
        return self._id

    @property
    def uri(self) -> str:
        return self._uri

    async def get_length(self) -> int:
        """Returns the length of the document in tokens."""
        text = await self.load_text()
        if len(text) <= 40000:
            return len(self._index._tokenizer.encode(text))
        else:
            return len(text) // 4  # Estimate for longer texts

    async def has_metadata(self) -> bool:
        """Checks if metadata file exists for the document."""
        metadata_path = Path(self.folder_path) / f"{self.id}.json"
        return await asyncio.to_thread(metadata_path.is_file)

    async def load_metadata(self) -> Dict[str, MetadataTypes]:
        """Loads metadata from disk if not already loaded."""
        if self._metadata is None:
            metadata_path = Path(self.folder_path) / f"{self.id}.json"
            try:
                json_str = await asyncio.to_thread(metadata_path.read_text)
                self._metadata = json.loads(json_str)
            except FileNotFoundError:
                raise FileNotFoundError(f"Metadata file for document '{self.uri}' not found.")
            except json.JSONDecodeError as err:
                raise ValueError(f"Error parsing metadata for document '{self.uri}': {err}")
            except Exception as err:
                raise RuntimeError(f"Error reading metadata for document '{self.uri}': {err}")
        return self._metadata

    async def load_text(self) -> str:
        """Loads document text from disk if not already loaded."""
        if self._text is None:
            text_path = Path(self.folder_path) / f"{self.id}.txt"
            try:
                self._text = await asyncio.to_thread(text_path.read_text)
            except FileNotFoundError:
                raise FileNotFoundError(f"Text file for document '{self.uri}' not found.")
            except Exception as err:
                raise RuntimeError(f"Error reading text file for document '{self.uri}': {err}")
        return self._text
