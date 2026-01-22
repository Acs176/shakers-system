from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class VectorStore(ABC):
    @abstractmethod
    def add(self, records: List[Dict]) -> int:
        raise NotImplementedError

    @abstractmethod
    def delete_by_object_key(self, object_key: str) -> int:
        raise NotImplementedError

    @abstractmethod
    def search(self, query: str, k: int = 3) -> List[Dict]:
        raise NotImplementedError

    @abstractmethod
    def save(self, out_dir: str):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, index_dir: str, model_name: Optional[str] = None):
        raise NotImplementedError
