from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import io
import struct
from typing import BinaryIO, Tuple, Optional
import numpy as np

Endian = str  # "<" little, ">" big, "=" native (standard), "@" native (unaligned)

@dataclass
class BinaryReader:
    path: Path | str
    endian: Endian = "<"
    _fh: Optional[BinaryIO] = None

    def __post_init__(self) -> None:
        self.path = Path(self.path)

    def __enter__(self) -> "BinaryReader":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def open(self) -> None:
        if self._fh is None:
            self._fh = open(self.path, "rb", buffering=0)

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None

    def tell(self) -> int:
        assert self._fh is not None
        return self._fh.tell()

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        assert self._fh is not None
        return self._fh.seek(offset, whence)

    def size(self) -> int:
        return self.path.stat().st_size

    def remaining(self) -> int:
        return self.size() - self.tell()

    def read(self, n: int) -> bytes:
        assert self._fh is not None
        b = self._fh.read(n)
        if len(b) != n:
            raise EOFError(f"Expected {n} bytes, got {len(b)} at offset {self.tell()}")
        return b

    def read_struct(self, fmt: str) -> Tuple:
        s = struct.Struct(self.endian + fmt)
        data = self.read(s.size)
        return s.unpack(data)

    def read_array(self, dtype: np.dtype, count: int) -> np.ndarray:
        dtype = np.dtype(dtype).newbyteorder(self.endian)
        nbytes = int(count) * dtype.itemsize
        buf = self.read(nbytes)
        return np.frombuffer(buf, dtype=dtype, count=count)

    def memmap(self, dtype: np.dtype, offset: int, shape: tuple[int, ...]) -> np.memmap:
        dtype = np.dtype(dtype).newbyteorder(self.endian)
        return np.memmap(self.path, mode="r", dtype=dtype, offset=offset, shape=shape)
