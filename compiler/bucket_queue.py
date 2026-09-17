from typing_extensions import Hashable

class BucketQueue[V: Hashable]:
    max_key: int
    buckets: list[dict[V, bool]]

    def __init__(self, max_key: int):
        self.max_key = max_key
        self.buckets = [{} for _ in range(max_key + 1)]

    def add(self, key: int, element: V):
        """Inserts the element into the specified bucket, ignoring duplicates"""
        self.buckets[key][element] = True

    def remove(self, key: int, element: V):
        """Removes the element from the given bucket"""
        self.buckets[key].pop(element, None)

    def update_key(self, new_key: int, old_key: int, element: V):
        """Removes the element from the given bucket, adding it to the new bucket"""
        self.remove(old_key, element)
        self.add(new_key, element)

    def get_min(self, max_allowed_key: int | None = None) -> tuple[V, int] | tuple[None, None]:
        """
        Returns the (item, key) with the lowest key, up to the max_allowed_key.
        Else, returns (None, None)
        """
        limit = self.max_key if max_allowed_key is None else max_allowed_key

        for k in range(limit + 1):
            if self.buckets[k]:
                item = next(iter(self.buckets[k]))
                return item, k
        return None, None

    # def pop_min(self, max_allowed_key: int | None = None) -> tuple[V, int] | tuple[None, None]:
    #     """
    #     Returns the (item, key) with the lowest key, up to the max_allowed_key and removes that item from the 
    #     Else, returns (None, None)
    #     """
    #     item, impact = self.get_min(max_allowed_key)
    #     if item is None or impact is None:
    #         return None, None 
    #     del self.buckets[impact][item]
    #     return item, impact


    def is_empty(self):
        return all(len(b) == 0 for b in self.buckets)

