from typing import List
import math


class ItemSelector:
    """
    A class for selecting items based on their similarity.
    """
    @staticmethod
    def cosine_similarity(vector1: List[float],
                          vector2: List[float]) -> float:
        """
        Returns the similarity between two vectors using the cosine similarity.
        """
        # the quotient of the dot product and the product of the norms
        return (ItemSelector.dot_product(vector1, vector2) /
                (ItemSelector.normalize(vector1) *
                 ItemSelector.normalize(vector2)))

    @staticmethod
    def normalize(vector: List[float]) -> float:
        """
        The norm of a vector is
            the square root of the sum of the squares of the elements.
        Returns the normalized value of a vector.
        """
        # crutch to santize lists of lists that come from some embedding models
        # this will almost certainly have consequences
        if isinstance(vector[0], list):
            vector = vector[0]
        # Initialize a variable to store the sum of the squares
        sum = 0
        # Loop through the elements of the array
        for i in range(len(vector)):
            # Square the element and add it to the sum
            sum += vector[i] * vector[i]
        # Return the square root of the sum
        return math.sqrt(sum)

    @staticmethod
    def normalized_cosine_similarity(vector1: List[float],
                                     norm1: float,
                                     vector2: List[float],
                                     norm2: float) -> float:
        """
        Returns the similarity between two vectors using the cosine similarity,
            considers norms.
        """
        # Return the quotient of the dot product and the product of the norms
        return ItemSelector.dot_product(vector1, vector2) / (norm1 * norm2)

    @staticmethod
    def select(metadata: dict,
               filter: dict) -> bool:
        """
        Handles filter logic.
        """
        if filter is None:
            return True
        for key in filter:
            if key == '$and':
                if not all(ItemSelector.select(metadata, f)
                           for f in filter['$and']):
                    return False
            elif key == '$or':
                if not any(ItemSelector.select(metadata, f)
                           for f in filter['$or']):
                    return False
            else:
                value = filter[key]
                if value is None:
                    return False
                elif isinstance(value, dict):
                    if not ItemSelector.metadataFilter(metadata.get(key),
                                                       value):
                        return False
                else:
                    if metadata.get(key) != value:
                        return False
        return True

    @staticmethod
    def dot_product(vector1: List[float],
                    vector2: List[float]) -> float:
        """
        Returns the dot product of two vectors.
        """
        # Zip the two vectors and multiply each pair, then sum the products
        if isinstance(vector1[0], list):
            vector1 = [item for sublist in vector1 for item in sublist]
        if isinstance(vector2[0], list):
            vector2 = [item for sublist in vector2 for item in sublist]
        
        return sum(a * b for a, b in zip(vector1, vector2))

    @staticmethod
    def metadata_filter(value,
                        filter) -> bool:
        """
        Handles metadata filter logic.
        """
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
                if not isinstance(value, float) or value <= filter[key]:
                    return False
            elif key == "$gte":
                if not isinstance(value, float) or value < filter[key]:
                    return False
            elif key == "$lt":
                if not isinstance(value, float) or value >= filter[key]:
                    return False
            elif key == "$lte":
                if not isinstance(value, float) or value > filter[key]:
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
