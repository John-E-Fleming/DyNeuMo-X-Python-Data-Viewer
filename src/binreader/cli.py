from __future__ import annotations
from pathlib import Path
import typer
import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console

from .dynx import read_sdio_logger, adc_to_uV
from .dsp import filter_offset, filter_band
from .spectrogram import multitaper_spectrogram

console = Console()
app = typer.Typer(add_completion=False, help="Binreader utilities")

def _run_pipeline(
    path: Path,
    file_name: str,
    num_channels: int,
    fs: float,
    gain: float,
    ch_hi: int,
    ch_lo: int,
    band_low: float,
    band_high: float,
    df: float,
    n: float,
    fmin: float,
    fmax: float,
    win_s: float,
    step_s: float,
    detrend: str,
    weighting: str,
    min_nfft: int,
    plot: bool,
    clim_percentiles: str,
    debug: bool,
) -> None:
    file_path = (path / file_name).resolve()
    try:
        console.print(f"[bold]Loading[/] {file_path} from {path.resolve()}")
        if not file_path.exists():
            console.print(f"[red]File not found:[/red] {file_path}")
            raise typer.Abort()

        file_bytes = file_path.stat().st_size
        console.print(f"Size: {file_bytes:,} bytes  |  Channels: {num_channels}  |  fs: {fs} Hz")

        ints = file_bytes // 4
        if ints % num_channels != 0:
            console.print(
                f"[red]File size not compatible with int32×channels[/red]\n"
                f"  int32 count = {ints:,}\n"
                f"  channels    = {num_channels}\n"
                f"  ints % channels = {ints % num_channels}\n"
                f"[yellow]Tip:[/yellow] Check --num-channels, file format, or endianness."
            )
            raise typer.Abort()

        X_adc = read_sdio_logger(path, file_name, num_channels=num_channels, use_memmap=True)
        X_uV = adc_to_uV(X_adc, gain=gain)
        raw = X_uV[ch_hi, :] - X_uV[ch_lo, :]

        console.print("[bold]Filtering[/] offset and band")
        x_off = filter_offset(raw)
        x_filt = filter_band(x_off, (band_low, band_high), fs)

        TW = n * df / 2.0
        L = max(int(np.floor(2 * TW) - 1), 1)

        console.print(f"[bold]Spectrogram[/] TW={TW:.3f}, L={L}, win={win_s}s, step={step_s}s, "
                      f"detrend={detrend}, weighting={weighting}, min_nfft={min_nfft}")
        S, freqs, times = multitaper_spectrogram(
            x_filt,
            fs=fs,
            frequency_range=(fmin, fmax),
            taper_params=(TW, L),
            window_params=(win_s, step_s),
            min_nfft=min_nfft,
            detrend_opt=detrend,
            weighting=weighting,
            verbose=False,
        )

        console.print(f"S: {S.shape} (F x T), freqs: {freqs[0]:.2f}-{freqs[-1]:.2f} Hz, frames: {len(times)}")

        if plot:
            Sdb = 10.0 * np.log10(np.where(S > 0, S, np.nan))
            p_lo, p_hi = [float(p.strip()) for p in clim_percentiles.split(",")]
            finite_vals = Sdb[np.isfinite(Sdb)]
            vmin = vmax = None
            if finite_vals.size:
                vmin, vmax = np.percentile(finite_vals, [p_lo, p_hi])

            plt.figure(figsize=(12, 4))
            extent = [times[0] / 3600.0, times[-1] / 3600.0, freqs[0], freqs[-1]]
            im = plt.imshow(Sdb, aspect="auto", origin="lower", extent=extent, vmin=vmin, vmax=vmax)
            plt.xlabel("Time (h)"); plt.ylabel("Frequency (Hz)")
            plt.title("Multitaper Spectrogram (dB)")
            plt.colorbar(im, label="Power (dB)")
            plt.tight_layout()
            plt.show()

    except Exception as e:
        if debug:
            raise
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Abort()

# ❌ No @app.callback here. The callback previously ran with defaults and aborted.
# ✅ Single subcommand only.
@app.command("dynx_pipeline")
def dynx_pipeline(
    path: Path = typer.Option(Path("."), help="Folder containing SdioLogger.bin"),
    file_name: str = typer.Option("SdioLogger.bin", help="Binary filename"),
    num_channels: int = typer.Option(9, help="Number of interleaved int32 channels"),
    fs: float = typer.Option(4000.0, help="Sampling rate (Hz)"),
    gain: float = typer.Option(24.0, help="Channel gain (x)"),
    ch_hi: int = typer.Option(6, help="High channel index (0-based)"),
    ch_lo: int = typer.Option(0, help="Low channel index (0-based)"),
    band_low: float = typer.Option(0.1, help="Bandpass low cutoff (Hz)"),
    band_high: float = typer.Option(40.0, help="Bandpass high cutoff (Hz)"),
    df: float = typer.Option(0.5, help="Target frequency resolution (Hz)"),
    n: float = typer.Option(15.0, help="Stationary window length for multitaper (s)"),
    fmin: float = typer.Option(0.0, help="Spectrogram min frequency (Hz)"),
    fmax: float = typer.Option(25.0, help="Spectrogram max frequency (Hz)"),
    win_s: float = typer.Option(30.0, help="Spectrogram window size (s)"),
    step_s: float = typer.Option(15.0, help="Spectrogram step size (s)"),
    detrend: str = typer.Option("constant", help="Detrend: 'linear'|'constant'|'off'"),
    weighting: str = typer.Option("unity", help="Taper weighting: 'unity'|'eigen'|'adapt'"),
    min_nfft: int = typer.Option(0, help="Minimum NFFT before nextpow2 padding"),
    plot: bool = typer.Option(True, help="Show multitaper spectrogram"),
    clim_percentiles: str = typer.Option("5,98", help="Color limits percentiles, e.g. '5,98'"),
    debug: bool = typer.Option(False, help="Print full traceback on error"),
):
    _run_pipeline(**locals())
