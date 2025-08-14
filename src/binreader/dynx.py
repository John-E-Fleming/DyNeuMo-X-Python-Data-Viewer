from __future__ import annotations
from pathlib import Path
import math
import numpy as np
from .reader import BinaryReader

def read_sdio_logger(
    path: str | Path,
    file_name: str,
    num_channels: int,
    endian: str = "<",
    use_memmap: bool = True,
) -> np.ndarray:
    """
    Read interleaved int32 ADC samples (channels x samples), like MATLAB fread([C,N],'int32').
    Uses np.memmap by default to avoid loading the whole file into RAM.
    """
    p = Path(path) / file_name
    file_bytes = p.stat().st_size
    bytes_per_sample = 4  # int32
    total_ints = file_bytes // bytes_per_sample

    if total_ints % num_channels != 0:
        raise ValueError(
            f"File size not divisible by 4*num_channels. "
            f"bytes={file_bytes}, int32s={total_ints}, channels={num_channels}."
        )

    num_samples = total_ints // num_channels
    dtype = np.dtype(endian + "i4")

    if use_memmap:
        mm = np.memmap(p, mode="r", dtype=dtype)
        # Column-major reshape to match MATLAB fread([C,N], ...) behavior
        return np.reshape(mm, (num_channels, num_samples), order="F")
    else:
        # Fallback: read into memory (not recommended for large files)
        with BinaryReader(p, endian=endian) as br:
            raw = br.read_array(dtype, count=total_ints)
        return raw.reshape((num_channels, num_samples), order="F")

def adc_to_uV(x: np.ndarray, gain: float) -> np.ndarray:
    """
    y = x * ((2*4.5)/gain) / (2^24) * 1e6
    Mirrors MATLAB: y = x * ((2.*4.5)./gain)./(2.^24).*1e6;
    """
    return x.astype(np.float64, copy=False) * ((2.0 * 4.5) / gain) / (2.0**24) * 1e6
