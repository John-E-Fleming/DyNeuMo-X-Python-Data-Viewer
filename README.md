# binreader – DyNeuMo‑X Binary Data Reader & Multitaper Spectrogram

`binreader` is a lightweight Python toolkit for reading **interleaved `int32` binary logs** (e.g., `SdioLogger.bin`), converting raw ADC counts to **microvolts**, applying basic EEG preprocessing (offset removal, band‑pass), and computing a **multitaper spectrogram** that closely mirrors a reference MATLAB implementation.

The code is organized as a small package with both a **CLI** and a **Python API**, designed for large recordings via zero‑copy `np.memmap`.

---

## Features

- Fast binary reader for **interleaved `int32`** (channels × samples) with MATLAB‑like column-major reshape.
- Device helpers for DyNeuMo‑X:
  - `read_sdio_logger(...)` → `(channels, samples)`
  - `adc_to_uV(...)` conversion (24‑bit ADC → µV) using `((2*4.5)/gain) / 2^24 * 1e6`
- Preprocessing:
  - `filter_offset(...)` (recursive high‑pass offset removal, MATLAB‑equivalent)
  - `filter_band(...)` (2nd‑order Butterworth band‑pass + zero‑phase `filtfilt`)
- **Multitaper spectrogram** (`DPSS` tapers) with:
  - `unity` / `eigen` / `adapt` weighting
  - `linear` / `constant` / `off` detrending
  - MATLAB‑matching NFFT rule and one‑sided PSD scaling
- Memory‑efficient: uses `np.memmap` to handle multi‑GB files.
- Clean **Typer** CLI with friendly validation and a `--debug` option.

> **Note:** The example pipeline computes a **bipolar channel**: `raw = ch_hi − ch_lo` (0‑based indices in Python). Defaults match the MATLAB script (`ch_hi=6`, `ch_lo=0` → “channel 7 minus channel 1” in MATLAB terms).

---

## Project structure

```
binaryio/
├── pyproject.toml
├── README.md
├── src/
│   └── binreader/
│       ├── __init__.py
│       ├── cli.py            # CLI (Typer): dynx_pipeline
│       ├── reader.py         # Generic binary reader helpers
│       ├── dynx.py           # DyNeuMo-X helpers (read_sdio_logger, adc_to_uV)
│       ├── dsp.py            # Filters (offset, bandpass, harmonics notch - optional)
│       └── spectrogram.py    # Multitaper spectrogram (MATLAB parity)
└── tests/
    ├── test_reader_basic.py
    └── data/
```

---

## Installation

### 1) Clone and install in editable mode

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e ".[dev]"
```

Your `pyproject.toml` should include (or similar):

```toml
[project]
dependencies = [
  "numpy>=1.26",
  "scipy>=1.12",
  "pandas>=2.2",
  "typer>=0.12",
  "rich>=13.7",
  "matplotlib>=3.9"
]

[project.optional-dependencies]
dev = ["pytest>=8", "mypy>=1.10", "ruff>=0.5"]
```

---

## CLI usage

The CLI currently exposes one command, **`dynx_pipeline`**, which reproduces the MATLAB workflow:

```bash
binreader dynx_pipeline \
  --path "/path/to/data" \
  --file-name "SdioLogger.bin" \
  --num-channels 9 \
  --fs 4000 --gain 24 \
  --ch-hi 6 --ch-lo 0 \
  --band-low 0.1 --band-high 40 \
  --df 0.5 --n 15 \
  --fmin 0 --fmax 25 \
  --win-s 30 --step-s 15 \
  --detrend constant --weighting unity \
  --min-nfft 0 --plot
```

### Windows PowerShell example
```powershell
binreader dynx_pipeline `
  --path "C:\...\DyNeuMo-X-Python-Data-Viewer\data" `
  --file-name "SdioLogger_miklos_night_2.bin" `
  --num-channels 9 --fs 4000 --gain 24 --ch-hi 6 --ch-lo 0 `
  --band-low 0.1 --band-high 40 `
  --df 0.5 --n 15 --fmin 0 --fmax 25 `
  --win-s 30 --step-s 15 `
  --detrend constant --weighting unity --min-nfft 0 --plot
```

### What you’ll see
- A console log with file path/size, parameters, and the spectrogram matrix shape: `S: (F x T)`.
- A Matplotlib figure:
  - X axis = time (hours)
  - Y axis = frequency (Hz) within `[fmin, fmax]`
  - Color = power (dB). The colormap limits are robustly set via `--clim-percentiles` (default `5,98`).

### Tips
- If you see **“File size not compatible with int32×channels”**, verify `--num-channels` and that the file truly contains interleaved `int32` (4 bytes per sample).
- Use `--no-plot` for quick dry runs or headless servers.
- Add `--debug` to surface a full Python traceback on errors.

---

## Python API

You can also call the components directly from Python:

```python
from pathlib import Path
import numpy as np
from binreader.dynx import read_sdio_logger, adc_to_uV
from binreader.dsp import filter_offset, filter_band
from binreader.spectrogram import multitaper_spectrogram

path = Path("data")
fname = "SdioLogger.bin"
X_adc = read_sdio_logger(path, fname, num_channels=9)     # (channels, samples) via memmap
X_uV = adc_to_uV(X_adc, gain=24.0)
raw = X_uV[6] - X_uV[0]

x_off = filter_offset(raw)
x_filt = filter_band(x_off, (0.1, 40.0), fs=4000.0)

df = 0.5; N = 15.0
TW = N * df / 2
L = max(int(np.floor(2*TW) - 1), 1)

S, freqs, times = multitaper_spectrogram(
    x_filt,
    fs=4000.0,
    frequency_range=(0.0, 25.0),
    taper_params=(TW, L),
    window_params=(30.0, 15.0),
    min_nfft=0,
    detrend_opt="constant",
    weighting="unity",
)
```

---

## Troubleshooting

- **“No such option: --N”** → The parameter is `--n` (lowercase).  
- **Missing command / unexpected extra argument** → Ensure your entry point is a Typer app exposing subcommands:  
  In `pyproject.toml` use `binreader = "binreader.cli:app"` and reinstall with `pip install -e .[dev]`.
- **Memmap vs RAM** → We use `np.memmap` to avoid loading entire files. If needed, disable via `use_memmap=False` in `read_sdio_logger` (not recommended for large logs).
- **Windows paths** → Quote paths containing spaces; prefer forward slashes in Bash; PowerShell requires backticks for line breaks.
- **Plot window doesn’t appear** → Try `--no-plot` to verify processing, or ensure a GUI backend is available (`matplotlib` may use `Agg` in headless envs).

---

## Notes on channel selection & accelerometer

- The pipeline computes a **bipolar** signal: `raw = X_uV[ch_hi] - X_uV[ch_lo]` with **0‑based** indices.
- The provided MATLAB script and this port **do not** decode accelerometer channels. If the log contains IMU axes, they’ll be treated as ADC counts and **µV** scaling will be wrong.  
  If you share the channel map and IMU sensitivity (LSB/g), we can add conversion to **g** and an IMU plotting command.

---

## Development

- **Run tests**: `pytest -q`
- **Lint/format**: `ruff check .` (and `ruff format .` if you use Ruff for formatting)
- **Type check**: `mypy src/`

Contributions welcome—PRs should include tests for new readers/record types.

---

## Acknowledgments

The multitaper spectrogram mirrors the MATLAB approach described in:
> Prerau et al., *Sleep Neurophysiological Dynamics Through the Lens of Multitaper Spectral Analysis*, Physiology, 2017.  
We implemented DPSS tapers and weighting schemes (`unity`, `eigen`, `adapt`) to align with that reference behavior.

---

## License

MIT (see `LICENSE`).

