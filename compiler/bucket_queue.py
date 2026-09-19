from typing_extensions import Hashable

class BucketQueue[V: Hashable]:
    max_key: int
    buckets: list[dict[V, None]]
    _min_key: int

    def __init__(self, max_key: int):
        self.max_key = max_key
        self.buckets = [dict() for _ in range(max_key + 1)]
        self._min_key = max_key + 1  # sentinel: beyond range means queue is empty

    def add(self, key: int, element: V):
        """Inserts the element into the specified bucket, ignoring duplicates"""
        self.buckets[key][element] = None
        if key < self._min_key:
            self._min_key = key

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
        # Advance _min_key lazily past empty buckets
        while self._min_key <= limit and not self.buckets[self._min_key]:
            self._min_key += 1
        if self._min_key > limit:
            return None, None
        item = next(iter(self.buckets[self._min_key]))
        return item, self._min_key

    def is_empty(self):
        return all(not b for b in self.buckets)
