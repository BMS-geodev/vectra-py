import os
import shutil
import json
from uuid import uuid4
from typing import List, Optional, Dict, Union, Any
from item_selector import ItemSelector
from custom_types import IndexItem, IndexStats, MetadataFilter, MetadataTypes, QueryResult


class CreateIndexConfig:
    def __init__(self, version: int, delete_if_exists: bool = False, metadata_config: Dict = {}):
        self.version = version
        self.delete_if_exists = delete_if_exists
        self.metadata_config = metadata_config


class LocalIndex:
    def __init__(self, folder_path: str, index_name: Optional[str] = None):
        self._folder_path = folder_path
        self._index_name = index_name or "index.json"
        self._data = None
        self._update = None

    @property
    def folder_path(self) -> str:
        return self._folder_path

    @property
    def index_name(self) -> str:
        return self._index_name

    async def begin_update(self) -> None:
        if self._update:
            raise ValueError('Update already in progress')

        await self.load_index_data()
        self._update = self._data.copy()

    def cancel_update(self) -> None:
        self._update = None

    async def create_index(self, config: CreateIndexConfig = CreateIndexConfig(version=1)) -> None:
        if self.is_index_created():
            if config.delete_if_exists:
                await self.delete_index()
            else:
                raise ValueError('Index already exists')
        try:
            os.mkdir(self._folder_path)
            self._data = {
                "version": config.version,
                "metadata_config": config.metadata_config,
                "items": []
            }
            with open(os.path.join(self._folder_path, self._index_name), 'w') as index_file:
                json.dump(self._data, index_file)
        except Exception:
            await self.delete_index()
            raise ValueError('Error creating index')

    async def delete_index(self) -> None:
        self._data = None
        try:
            shutil.rmtree(self._folder_path)
        except Exception as err:
            print(err)

    async def delete_item(self, id: str) -> None:
        if self._update:
            index = next((i for i, item in enumerate(self._update["items"]) if item["id"] == id), None)
            if index is not None:
                self._update["items"].pop(index)
        else:
            await self.begin_update()
            index = next((i for i, item in enumerate(self._update["items"]) if item["id"] == id), None)
            if index is not None:
                self._update["items"].pop(index)
            await self.end_update()

    async def end_update(self) -> None:
        if not self._update:
            raise ValueError('No update in progress')

        try:
            with open(os.path.join(self._folder_path, self._index_name), 'w') as index_file:
                json.dump(self._update, index_file)
            self._data = self._update.copy()
            self._update = None
        except Exception as err:
            raise ValueError(f'Error saving index: {str(err)}')

    async def get_index_stats(self) -> IndexStats:
        await self.load_index_data()
        return {
            "version": self._data["version"],
            "metadata_config": self._data["metadata_config"],
            "items": len(self._data["items"])
        }

    async def get_item(self, id: str) -> Optional[IndexItem]:
        await self.load_index_data()
        item = next((item for item in self._data["items"] if item["id"] == id), None)
        return item

    async def insert_item(self, item: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if self._update:
            return await self.add_item_to_update(item, True)
        else:
            await self.begin_update()
            new_item = await self.add_item_to_update(item, True)
            await self.end_update()
            return new_item

    def is_index_created(self) -> bool:
        return os.path.exists(os.path.join(self._folder_path, self._index_name))

    async def list_items(self) -> List[IndexItem]:
        await self.load_index_data()
        return self._data["items"][:]

    async def list_items_by_metadata(self, filter: MetadataFilter) -> List[IndexItem]:
        await self.load_index_data()
        return [item for item in self._data["items"] if ItemSelector.select(item["metadata"], filter)]

    async def query_items(self,
                          vector: List[float],
                          top_k: int,
                          filter: Optional[MetadataFilter] = None) -> List[QueryResult]:
        await self.load_index_data()

        items = self._data["items"][:]
        if filter:
            items = [item for item in items if ItemSelector.select(item["metadata"], filter)]

        norm = ItemSelector.normalize(vector)
        distances = []
        for i, item in enumerate(items):
            distance = ItemSelector.normalized_cosine_similarity(vector, norm, item["vector"], item["norm"])
            distances.append({"index": i, "distance": distance})

        distances.sort(key=lambda x: x["distance"], reverse=True)
        top_items = distances[:top_k]

        for item in top_items:
            if "metadataFile" in items[item["index"]]:
                metadata_path = os.path.join(self._folder_path, items[item["index"]]["metadataFile"])
                with open(metadata_path, 'r') as metadata_file:
                    items[item["index"]]["metadata"] = json.load(metadata_file)

        return [{"item": items[item["index"]], "score": item["distance"]} for item in top_items]

    async def upsert_item(self, item: Optional[Dict[str, Any]] = None) -> IndexItem:
        if self._update:
            return await self.add_item_to_update(item, False)
        else:
            await self.begin_update()
            new_item = await self.add_item_to_update(item, False)
            await self.end_update()
            return new_item

    async def load_index_data(self) -> None:
        if self._data:
            return

        if not self.is_index_created():
            raise ValueError('Index does not exist')

        try:
            with open(os.path.join(self._folder_path, self._index_name), 'r') as index_file:
                self._data = json.load(index_file)
        except Exception:
            raise ValueError('Error loading index data')

    async def add_item_to_update(self, item: Optional[Dict[str, Any]], unique: bool) -> IndexItem:
        if "vector" not in item:
            raise ValueError('Vector is required')

        item_id = item.get("id") or str(uuid4())
        if unique:
            existing_item = next((i for i in self._update["items"] if i["id"] == item_id), None)
            if existing_item:
                raise ValueError(f'Item with id {item_id} already exists')

        metadata = {}
        metadata_file = None
        if (
            "metadata" in item
            and self._update["metadata_config"].get("indexed")
            and len(self._update["metadata_config"]["indexed"]) > 0
        ):
            for key in self._update["metadata_config"]["indexed"]:
                if key in item["metadata"]:
                    metadata[key] = item["metadata"][key]
            if item.get("metadata"):
                metadata_file = f'{str(uuid4())}.json'
                metadata_path = os.path.join(self._folder_path, metadata_file)
                with open(metadata_path, 'w') as metadata_file:
                    json.dump(item["metadata"], metadata_file)
        elif item.get("metadata"):
            metadata = item["metadata"]
        # print('local index, after metadata')
        # print('item vector type and len', type(item["vector"]), len(item["vector"]))
        # print('item vector chunk inspection', item["vector"][0])
        try:
            new_item = {
                "id": item_id,
                "metadata": metadata,
                "vector": item["vector"],
                "norm": ItemSelector.normalize(item["vector"])
            }
        except Exception as e:
            raise ValueError(f'Error creating item: {e}')
        if metadata_file:
            new_item["metadataFile"] = metadata_file

        if not unique:
            existing_item = next((i for i in self._update["items"] if i["id"] == item_id), None)
            if existing_item:
                existing_item.update(new_item)
                return existing_item

        self._update["items"].append(new_item)
        return new_item
