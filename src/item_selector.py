from typing import List
import math

class ItemSelector:
    @staticmethod
    def cosineSimilarity(vector1: List[int], vector2: List[int]) -> float:
        # Return the quotient of the dot product and the product of the norms
        return ItemSelector.dotProduct(vector1, vector2) / (ItemSelector.normalize(vector1) * ItemSelector.normalize(vector2))

    @staticmethod
    def normalize(vector: List[int]) -> float:
        # Initialize a variable to store the sum of the squares
        sum = 0
        # Loop through the elements of the array
        for i in range(len(vector)):
            # Square the element and add it to the sum
            sum += vector[i] * vector[i]
        # Return the square root of the sum
        return math.sqrt(sum)
    
    @staticmethod
    def normalizedCosineSimilarity(vector1: List[int], norm1: float, vector2: List[int], norm2: float) -> float:
        # Return the quotient of the dot product and the product of the norms
        return ItemSelector.dotProduct(vector1, vector2) / (norm1 * norm2)
    
    @staticmethod
    def select(metadata: dict, filter: dict) -> bool:
        if filter is None:
            return True
        for key in filter:
            if key == '$and':
                if not all(ItemSelector.select(metadata, f) for f in filter['$and']):
                    return False
            elif key == '$or':
                if not any(ItemSelector.select(metadata, f) for f in filter['$or']):
                    return False
            else:
                value = filter[key]
                if value is None:
                    return False
                elif isinstance(value, dict):
                    if not ItemSelector.metadataFilter(metadata.get(key), value):
                        return False
                else:
                    if metadata.get(key) != value:
                        return False
        return True
    
    @staticmethod
    def dotProduct(vector1: List[int], vector2: List[int]) -> int:
        # Zip the two vectors and multiply each pair, then sum the products
        return sum(a * b for a, b in zip(vector1, vector2))
    
    @staticmethod
    def metadata_filter(value, filter):
        if value is None:
            return False
        
        for key in filter:
            if key == "$eq":
                if value != filter[key]:
                    return False
            elif key == "$ne":
                if value == filter[key]:
                    return False
            elif key == "$gt":
                if not isinstance(value, int) or value <= filter[key]:
                    return False
            elif key == "$gte":
                if not isinstance(value, int) or value < filter[key]:
                    return False
            elif key == "$lt":
                if not isinstance(value, int) or value >= filter[key]:
                    return False
            elif key == "$lte":
                if not isinstance(value, int) or value > filter[key]:
                    return False
            elif key == "$in":
                if not isinstance(value, bool) or value not in filter[key]:
                    return False
            elif key == "$nin":
                if not isinstance(value, bool) or value in filter[key]:
                    return False
            else:
                if value != filter[key]:
                    return False
        
        return True