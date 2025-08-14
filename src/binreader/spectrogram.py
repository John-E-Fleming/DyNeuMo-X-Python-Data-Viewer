from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Literal, Tuple
from scipy.signal.windows import dpss
from scipy.signal import detrend as sp_detrend

def _next_pow2_int(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (int(np.ceil(np.log2(n))))

def multitaper_spectrogram(
    data: NDArray[np.floating],
    fs: float,
    frequency_range: Tuple[float, float] | None = None,  # (fmin, fmax)
    taper_params: Tuple[float, int] = (5.0, 9),          # (time_halfbandwidth, num_tapers)
    window_params: Tuple[float, float] = (5.0, 1.0),     # (win_s, step_s)
    min_nfft: int = 0,
    detrend_opt: Literal["linear", "constant", "off"] | bool = "linear",
    weighting: Literal["unity", "eigen", "adapt"] = "unity",
    verbose: bool = False,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """
    Multitaper spectrogram with MATLAB-compatible behavior:

    * Windows: size/step in seconds -> samples (rounded to nearest int like MATLAB warning path).
    * NFFT: max( 2^nextpow2(win), win, 2^nextpow2(min_nfft) ).
    * Detrend: 'linear' / 'constant' / 'off' (bool True treated as 'linear' for parity).
    * Weighting: 'unity' (equal), 'eigen' (eigenvalue / num_tapers), 'adapt' (Percival & Walden, 3 iters).
    * One-sided PSD: double non-DC/non-Nyquist bins, then divide by Fs.
    * Output shapes: S (F x T), times (T,), freqs (F,). Frequencies limited to `frequency_range`.

    Returns
    -------
    S : ndarray, shape (F, T)
        Power spectral density.
    times : ndarray, shape (T,)
        Center time (s) of each window.
    freqs : ndarray, shape (F,)
        One-sided frequency bins (Hz) within frequency_range.
    """
    x = np.asarray(data, dtype=np.float64).reshape(-1)  # column vector behavior
    N = x.size
    if frequency_range is None:
        frequency_range = (0.0, fs / 2.0)
    fmin, fmax = float(frequency_range[0]), float(frequency_range[1])
    if fmax > fs / 2.0:
        fmax = fs / 2.0  # clamp like MATLAB

    # --- window sizes in samples (MATLAB rounds and warns; we just round) ---
    win_s, step_s = window_params
    win_n = int(round(win_s * fs))
    hop_n = int(round(step_s * fs))
    if win_n <= 0 or hop_n <= 0:
        raise ValueError("Window size and step must be > 0.")

    # --- NFFT per MATLAB rule ---
    nfft = max(_next_pow2_int(win_n), win_n, _next_pow2_int(min_nfft))

    # --- DPSS tapers and eigenvalues ---
    TW, K = taper_params
    if K < 1:
        raise ValueError("num_tapers must be >= 1")
    tapers, eigvals = dpss(M=win_n, NW=TW, Kmax=K, return_ratios=True)  # (K, win_n)

    # --- weighting vector or adaptive flag ---
    if weighting == "unity":
        w = np.ones(K, dtype=np.float64) / K
        use_adapt = False
    elif weighting == "eigen":
        # MATLAB: wt = DPSS_eigen / num_tapers (NOT normalized to sum=1)
        w = eigvals / K
        use_adapt = False
    elif weighting == "adapt":
        w = None
        use_adapt = True
    else:
        raise ValueError("weighting must be 'unity', 'eigen', or 'adapt'")

    # --- rFFT frequency axis (one-sided bins: 0..Fs/2 inclusive) ---
    M = nfft // 2 + 1
    freqs = np.linspace(0.0, fs / 2.0, M)

    # Limit to frequency_range AFTER we build PSD (MATLAB does the same)
    fmask = (freqs >= fmin) & (freqs <= fmax)
    keep_idx = np.where(fmask)[0]
    F = keep_idx.size

    # --- window starts (0-based) and centers ---
    starts = np.arange(0, N - win_n + 1, hop_n, dtype=int)
    T = starts.size
    times = (starts + (win_n // 2)) / fs

    # --- allocate output ---
    S = np.zeros((F, T), dtype=np.float64)

    # --- detrend option (MATLAB allows true/false; true ~ 'linear') ---
    if detrend_opt is True:
        detrend_opt = "linear"
    if detrend_opt is False or detrend_opt == "off":
        detrend_mode: str | None = None
    elif detrend_opt in ("linear", "constant"):
        detrend_mode = detrend_opt
    else:
        raise ValueError("detrend_opt must be 'linear', 'constant', 'off'/False, or True")

    # --- main loop over frames ---
    for ti, s0 in enumerate(starts):
        seg = x[s0 : s0 + win_n].copy()
        # Skip empty or all-NaN segments like MATLAB behavior
        if not np.any(seg) or np.all(np.isnan(seg)):
            # leave zeros (or NaNs if requested)
            if np.any(np.isnan(seg)):
                S[:, ti] = np.nan
            continue

        if detrend_mode is not None:
            seg = sp_detrend(seg, type=detrend_mode)

        # (K, win_n) * (win_n,) -> (K, win_n)
        tapered = tapers * seg[None, :]

        # rFFT across time axis -> (K, M), power = |FFT|^2
        fft = np.fft.rfft(tapered, n=nfft, axis=1)
        Spower = (fft.real**2 + fft.imag**2)  # (K, M)

        if use_adapt:
            # Percival & Walden adaptive weights (3 iters)
            Tpower = float(np.dot(seg, seg) / len(seg))  # x'*x / N
            if K >= 2:
                Sp_iter = np.mean(Spower[:2, :], axis=0)  # (M,)
            else:
                Sp_iter = Spower[0, :].copy()
            a = (1.0 - eigvals) * Tpower  # (K,)

            for _ in range(3):
                # denom: (K, M) = Sp_iter * eig + a
                denom = (Sp_iter[None, :] * eigvals[:, None]) + (a[:, None])
                b = Sp_iter[None, :] / np.maximum(denom, 1e-30)     # (K, M)
                wk = (b**2) * (eigvals[:, None])                    # (K, M)
                num = np.sum(wk * Spower, axis=0)                   # (M,)
                den = np.sum(wk, axis=0)                            # (M,)
                Sp_iter = num / np.maximum(den, 1e-30)
            mt_spec = Sp_iter  # (M,)
        else:
            # eigen/unity weights across tapers
            # Spower: (K, M), w: (K,) -> (M,)
            mt_spec = Spower.T @ w

        # --- one-sided PSD scaling like MATLAB ---
        # double non-DC/non-Nyquist bins
        mt_sel = mt_spec[keep_idx]  # (F,)
        sf = freqs[keep_idx]
        # factor=2 except for f=0 or f=Fs/2 (if present)
        factor = np.ones_like(mt_sel)
        factor[(sf != 0.0) & ~np.isclose(sf, fs / 2.0)] = 2.0
        mt_psd = (mt_sel * factor) / fs  # divide by Fs at the end

        S[:, ti] = mt_psd

    if verbose:
        df = fs / nfft
        print()
        print("Multitaper Spectrogram Properties:")
        print(f"    Spectral Resolution: {(2*TW)/ (win_n/fs):.6g} Hz")
        print(f"    Window Length: {win_n/fs:.6g} s")
        print(f"    Window Step: {hop_n/fs:.6g} s")
        print(f"    Time Half-Bandwidth Product: {TW}")
        print(f"    Number of Tapers: {K}")
        print(f"    Frequency Range: {fmin} Hz - {fmax} Hz")
        print(f"    Detrending: {detrend_opt if detrend_mode else 'Off'}")
        print()

    return S, freqs[keep_idx], times
