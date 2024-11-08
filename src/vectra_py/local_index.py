import os
import asyncio
import json
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from .custom_types import IndexItem, IndexStats, MetadataFilter, MetadataTypes, QueryResult
from .item_selector import ItemSelector


@dataclass
class CreateIndexConfig:
    version: int = 1
    delete_if_exists: bool = False
    metadata_config: Dict[str, List[str]] = field(default_factory=dict)


class LocalIndex:
    def __init__(self, folder_path: str, index_name: str = "index.json"):
        self._folder_path = Path(folder_path)
        self._index_name = index_name
        self._data: Optional[Dict[str, Any]] = None
        self._update: Optional[Dict[str, Any]] = None

    async def init_index(self):
        """Ensure the index is created with default fields if it does not exist or is invalid."""
        if not await self.is_index_created() or self._is_invalid_index_file():
            config = CreateIndexConfig(version=1, delete_if_exists=True)
            await self.create_index(config)
            print(f"Index created with default fields at {self._folder_path / self._index_name}")
        else:
            await self._load_index_data()

    def _is_invalid_index_file(self) -> bool:
        """Check if index.json is invalid (e.g., contains 'null' or invalid JSON)."""
        index_file = self._folder_path / self._index_name
        try:
            with open(index_file, "r") as file:
                data = json.load(file)
                return data is None  # Returns `True` if the file contains `null`
        except json.JSONDecodeError:
            return True  # Treats invalid JSON as an invalid file

    @property
    def folder_path(self) -> str:
        return str(self._folder_path)

    @property
    def index_name(self) -> str:
        return self._index_name

    async def begin_update(self) -> None:
        if self._update is not None:
            raise RuntimeError("Update already in progress")
        await self._load_index_data()
        if self._data is None:
            raise RuntimeError("Failed to load index data for update.")
        self._update = self._data.copy()

    def cancel_update(self) -> None:
        self._update = None

    async def create_index(self, config: CreateIndexConfig = CreateIndexConfig()) -> None:
        if await self.is_index_created():
            if config.delete_if_exists:
                await self.delete_index()
            else:
                raise RuntimeError("Index already exists")

        try:
            self._folder_path.mkdir(parents=True, exist_ok=True)
            self._data = {
                "version": config.version,
                "metadata_config": config.metadata_config,
                "items": []
            }
            await self._save_index()
        except Exception:
            await self.delete_index()
            raise RuntimeError("Error creating index")

    async def delete_index(self) -> None:
        self._data = None
        for item in self._folder_path.iterdir():
            item.unlink()
        self._folder_path.rmdir()

    async def delete_item(self, item_id: str) -> None:
        await self.begin_update()
        self._update["items"] = [item for item in self._update["items"] if item["id"] != item_id]
        await self.end_update()

    async def end_update(self) -> None:
        if self._update is None:
            raise RuntimeError("No update in progress")
        await self._save_index()
        self._data = self._update
        self._update = None

    async def get_index_stats(self) -> IndexStats:
        await self._load_index_data()
        return IndexStats(
            version=self._data["version"],
            metadata_config=self._data["metadata_config"],
            items=len(self._data["items"])
        )

    async def get_item(self, item_id: str) -> Optional[IndexItem]:
        await self._load_index_data()
        return next((item for item in self._data["items"] if item["id"] == item_id), None)

    async def insert_item(self, item: Dict[str, Any], manage_update: bool = True) -> IndexItem:
        if manage_update:
            await self.begin_update()
        new_item = await self._add_item_to_update(item, unique=True)
        if manage_update:
            await self.end_update()
        return new_item

    async def is_index_created(self) -> bool:
        # index_path = self._folder_path / self._index_name
        # exists = index_path.exists()
        # print(f"is_index_created check: {index_path} exists? {exists}")
        # return exists
        return (self._folder_path / self._index_name).exists()

    async def list_items(self) -> List[IndexItem]:
        await self._load_index_data()
        return self._data["items"].copy()

    async def list_items_by_metadata(self, filter: MetadataFilter) -> List[IndexItem]:
        await self._load_index_data()
        return [item for item in self._data["items"] if ItemSelector.select(item["metadata"], filter)]

    async def query_items(self, vector: List[float], top_k: int, filter: Optional[MetadataFilter] = None) -> List[QueryResult]:
        await self._load_index_data()
        items = [item for item in self._data["items"] if not filter or ItemSelector.select(item["metadata"], filter)]
        norm_vector = ItemSelector.normalize(vector)

        distances = [
            {"index": i, "distance": ItemSelector.normalized_cosine_similarity(vector, norm_vector, item["vector"], item["norm"])}
            for i, item in enumerate(items)
        ]
        distances.sort(key=lambda x: -x["distance"])
        top_items = [items[d["index"]] for d in distances[:top_k]]

        for item in top_items:
            if "metadataFile" in item:
                with open(self._folder_path / item["metadataFile"], "r") as metadata_file:
                    item["metadata"] = json.load(metadata_file)
        return [{"item": item, "score": d["distance"]} for item, d in zip(top_items, distances[:top_k])]

    async def upsert_item(self, item: Dict[str, Any]) -> IndexItem:
        await self.begin_update()
        updated_item = await self._add_item_to_update(item, unique=False)
        await self.end_update()
        return updated_item

    async def _load_index_data(self) -> None:
        if self._data is None:
            if not await self.is_index_created():
                raise RuntimeError("Index does not exist")
            with open(self._folder_path / self._index_name, "r") as file:
                self._data = json.load(file)

    async def _save_index(self) -> None:
        with open(self._folder_path / self._index_name, "w") as file:
            json.dump(self._update, file)

    async def _add_item_to_update(self, item: Dict[str, Any], unique: bool) -> IndexItem:
        if "vector" not in item:
            raise ValueError("Vector is required")

        item_id = item.get("id", str(uuid.uuid4()))
        if unique and any(i["id"] == item_id for i in self._update["items"]):
            raise RuntimeError(f"Item with id {item_id} already exists")

        metadata = {}
        metadata_file = None
        if "indexed" in self._update["metadata_config"] and item.get("metadata"):
            metadata = {key: item["metadata"][key] for key in self._update["metadata_config"]["indexed"] if key in item["metadata"]}
            metadata_file = f"{uuid.uuid4()}.json"
            with open(self._folder_path / metadata_file, "w") as file:
                json.dump(item["metadata"], file)

        new_item = IndexItem(
            id=item_id,
            metadata=metadata or item.get("metadata"),
            vector=item["vector"],
            norm=ItemSelector.normalize(item["vector"])
        )
        if metadata_file:
            new_item.metadataFile = metadata_file

        existing = next((i for i in self._update["items"] if i["id"] == item_id), None)
        if existing:
            existing.update(new_item.__dict__)
        else:
            self._update["items"].append(new_item.__dict__)
        return new_item
