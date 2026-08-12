import threading
from collections import OrderedDict
from typing import Hashable

class SimpleCacheStore:
    """Thread-safe in-memory cache store."""
    def __init__(self, max_size=10000):
        self.store = OrderedDict()
        self.max_size = max_size
        self._lock = threading.RLock()
    
    def add(self, key: Hashable, result):
        with self._lock:
            if len(self.store) >= self.max_size:
                self.store.popitem(last=False)
            self.store[key] = result
    
    def contains(self, key: Hashable) -> bool:
        with self._lock:
            return key in self.store
    
    def get(self, key: Hashable):
        with self._lock:
            self.store.move_to_end(key)
            return self.store.get(key)

    def clear(self):
        """Remove all entries from the cache."""
        with self._lock:
            self.store.clear()
