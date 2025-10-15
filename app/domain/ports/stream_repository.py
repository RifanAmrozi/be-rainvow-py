from abc import ABC, abstractmethod
from typing import Iterator


class StreamRepositoryPort(ABC):
@abstractmethod
def frames(self, rtsp_url: str) -> Iterator[bytes]:
"""Yield JPEG-encoded frames as raw bytes (one frame per yield)."""
raise NotImplementedError