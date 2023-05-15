from typing import Any, Dict, List, Optional, Union
import json
import os
import asyncio
import uuid

from src.item_selector import ItemSelector

class CreateIndexConfig:
    def __init__(self, version: int, deleteIfExists: bool = False, metadata_config: dict = {}):
        self.version = version
        self.deleteIfExists = deleteIfExists
        self.metadata_config = metadata_config

class IndexStats:
    def __init__(self, version: int, metadata_config: dict, items: int):
        self.version = version
        self.metadata_config = metadata_config
        self.items = items

class IndexItem:
    def __init__(self, id: str, metadata: dict, vector: List[int], norm: int):
        self.id = id
        self.metadata = metadata
        self.vector = vector
        self.norm = norm
        self.metadataFile = None

class QueryResult:
    def __init__(self, item: IndexItem, score: int):
        self.item = item
        self.score = score

MetadataTypes = Union[int, str, bool]

class MetadataFilter:
    def __init__(self):
        self.filter_dict = {}
    def _update_filter_dict(self, filter_type, value):
        if filter_type not in self.filter_dict:
            self.filter_dict[filter_type] = value
        else:
            raise ValueError(f"{filter_type} operator already exists in filter.")
    def eq(self, value: MetadataTypes):
        self._update_filter_dict('$eq', value)
        return self
    
    def ne(self, value: MetadataTypes):
        self._update_filter_dict('$ne', value)
        return self
    
    def gt(self, value: int):
        self._update_filter_dict('$gt', value)
        return self
    
    def gte(self, value: int):
        self._update_filter_dict('$gte', value)
        return self
    
    def lt(self, value: int):
        self._update_filter_dict('$lt', value)
        return self
    
    def lte(self, value: int):
        self._update_filter_dict('$lte', value)
        return self
    
    def in_array(self, values: List[Union[int, str]]):
        self._update_filter_dict('$in', values)
        return self
    
    def not_in_array(self, values: List[Union[int, str]]):
        self._update_filter_dict('$nin', values)
        return self
    
    def and_filter(self, filters: List):
        self._update_filter_dict('$and', filters)
        return self
    
    def or_filter(self, filters: List):
        self._update_filter_dict('$or', filters)
        return self

class LocalIndex:
    def __init__(self, folderPath: str):
        self._folderPath = folderPath
        self._data = None
        self._update = None

    async def beginUpdate(self) -> None:
        if self._update:
            raise Exception('Update already in progress')
        await self.loadIndexData()
        self._update = self._data.copy()

    def cancelUpdate(self) -> None:
        self._update = None

    def createIndex(self, config: Dict[str, Any] = {'version':1}) -> None:
        if self.isIndexCreated():
            if config.get('deleteIfExists'):
                self.deleteIndex()
            else:
                raise Exception('Index already exists')
        try:
            os.makedirs(self._folderPath, exist_ok=True)
            self._data = {
                'version': config['version'],
                'metadata_config': config.get('metadata_config', {}),
                'items': []
            }
            with open(os.path.join(self._folderPath, 'index.json'), 'w', encoding='utf-8') as f:
                json.dump(self._data, f)
        except Exception:
            self.deleteIndex()
            raise Exception('Error creating index')
    
    async def deleteIndex(self) -> None:
        self._data = None
        await asyncio.create_subprocess_shell(f'rm -rf {self._folderPath}',
                                               stderr=asyncio.subprocess.PIPE,
                                               stdout=asyncio.subprocess.PIPE)
    
    async def deleteItem(self, id: str) -> None:
        if self._update:
            index = next((i for i, item in enumerate(self._update['items']) if item['id'] == id), None)
            if index is not None:
                self._update['items'].pop(index)
        else:
            await self.beginUpdate()
            index = next((i for i, item in enumerate(self._update['items']) if item['id'] == id), None)
            if index is not None:
                self._update['items'].pop(index)
            await self.endUpdate()
    
    async def endUpdate(self) -> None:
        if not self._update:
            raise Exception('No update in progress')
        try:
            with open(os.path.join(self._folderPath, 'index.json'), 'w', encoding='utf-8') as f:
                json.dump(self._update, f)
            self._data = self._update
            self._update = None
        except Exception as e:
            raise Exception(f'Error saving index: {str(e)}')
    
    async def getIndexStats(self) -> Dict[str, Union[int, Dict[str, Any]]]:
        await self.loadIndexData()
        return {
            'version': self._data['version'],
            'metadata_config': self._data['metadata_config'],
            'items': len(self._data['items'])
        }
    
    async def getItem(self, id: str) -> Optional[Dict[str, Any]]:
        await self.loadIndexData()
        return next((item for item in self._data['items'] if item['id'] == id), None)
    
    async def insertItem(self, item: Dict[str, Any]) -> Dict[str, Any]:
        if self._update:
            return await self.addItemToUpdate(item, True)
        else:
            await self.beginUpdate()
            new_item = await self.addItemToUpdate(item, True)
            await self.endUpdate()
            return new_item
    
    def isIndexCreated(self) -> bool:
        return os.path.exists(os.path.join(self._folderPath, 'index.json'))
    
    async def listItems(self) -> List[Dict[str, Any]]:
        await self.loadIndexData()
        return self._data['items'].copy()
    
    async def listItemsByMetadata(self, filter: Dict[str, Any]) -> List[Dict[str, Any]]:
        await self.loadIndexData()
        return [i for i in self._data['items'] if ItemSelector.select(i['metadata'], filter)]
    
    async def queryItems(self, vector: List[float], topK: int, filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        await self.loadIndexData()
        # Filter items
        items = self._data['items']
        if filter:
            items = [i for i in items if ItemSelector.select(i['metadata'], filter)]
        # Calculate distances
        norm = ItemSelector.normalize(vector)
        distances = []
        for i, item in enumerate(items):
            distance = ItemSelector.normalizedCosineSimilarity(vector, norm, item['vector'], item['norm'])
            distances.append({'index': i, 'distance': distance})
        # Sort by distance DESCENDING
        distances = sorted(distances, key=lambda k: k['distance'], reverse=True)
        # Find top k
        top = []
        for d in distances[:topK]:
            top.append({
                'item': dict(items[d['index']]),
                'score': d['distance']
            })
        # Load external metadata
        for item in top:
            if item['item'].get('metadataFile', None):
                metadata_path = os.path.join(self._folderPath, item['item']['metadataFile'])
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    item['item']['metadata'] = metadata
        return top
    
    async def upsertItem(self, item: Dict[str, Any]) -> Dict[str, Any]:
        if self._update:
            return await self.addItemToUpdate(item, False)
        else:
            await self.beginUpdate()
            new_item = await self.addItemToUpdate(item, False)
            await self.endUpdate()
            return new_item
    
    async def loadIndexData(self) -> None:
        if self._data:
            return
        if not self.isIndexCreated():
            raise Exception('Index does not exist')
        with open(os.path.join(self._folderPath, 'index.json'), 'r', encoding='utf-8') as f:
            self._data = json.load(f)

    async def addItemToUpdate(self, item: dict, unique: bool) -> dict:
        # Ensure vector is provided
        if 'vector' not in item:
            raise ValueError('Vector is required')
        
        # Ensure unique
        id = item.get('id', uuid.uuid4())
        if unique:
            existing = next((i for i in self._update['items'] if i['id'] == id), None)
            if existing:
                raise ValueError(f'Item with id {id} already exists')
        
        # Check for indexed metadata
        metadata = {}
        metadata_file = None
        if self._update['metadata_config'].get('indexed') and len(self._update['metadata_config']['indexed']) > 0 and 'metadata' in item:
            # Copy only indexed metadata
            for key in self._update['metadata_config']['indexed']:
                if item['metadata'] and item['metadata'].get(key):
                    metadata[key] = item['metadata'][key]
            
            # Save remaining metadata to disk
            metadata_file = f"{str(uuid.uuid4())}.json"
            metadata_path = os.path.join(self._folderPath, metadata_file)
            with open(metadata_path, 'w') as f:
                json.dump(item['metadata'], f)
        elif 'metadata' in item:
            metadata = item['metadata']
        
        # Create new item
        new_item = {
            'id': str(id),
            'metadata': metadata,
            'vector': item['vector'],
            'norm': ItemSelector.normalize(item['vector'])
        }
        if metadata_file:
            new_item['metadataFile'] = metadata_file
        # Add item to index
        id = new_item['id']

        if not unique:
            existing = next((i for i in self._update['items'] if i['id'] == id), None)
            if existing:
                existing['metadata'] = new_item['metadata']
                existing['vector'] = new_item['vector']
                existing['metadataFile'] = new_item.get('metadataFile')
                return existing
            else:
                self._update['items'].append(new_item)
                return new_item
        else:
            self._update['items'].append(new_item)
            return new_item

class IndexData:
    def __init__(self, version: int, metadata_config: dict, items: List[dict]):
        self.version = version
        self.metadata_config = metadata_config
        self.items = items