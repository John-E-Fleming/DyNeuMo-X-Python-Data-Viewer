from __future__ import annotations
import numpy as np
from scipy.signal import butter, filtfilt

def filter_offset(x: np.ndarray) -> np.ndarray:
    """
    Replicates MATLAB:
    x = [x(:,1), x]; y(:,i) = 0.996 * (y(:,i-1) + x(:,i) - x(:,i-1))
    Works with 1D or 2D arrays. If 2D, expects shape (samples,) or (channels, samples).
    """
    x = np.asarray(x)
    if x.ndim == 1:
        x_pad = np.concatenate([x[:1], x], axis=0)[None, :]  # (1, N+1)
        unflip = True
    elif x.ndim == 2:
        # assume (samples, )? we want (channels, samples)
        if x.shape[0] > x.shape[1]:
            # input likely (samples, channels); transpose to (channels, samples)
            x = x.T
        x_pad = np.concatenate([x[:, :1], x], axis=1)        # (C, N+1)
        unflip = False
    else:
        raise ValueError("x must be 1D or 2D")

    y = np.zeros_like(x_pad, dtype=np.float64)
    for i in range(1, x_pad.shape[1]):
        y[:, i] = 0.996 * (y[:, i - 1] + x_pad[:, i] - x_pad[:, i - 1])

    y = y[:, 1:]  # drop the pad
    if x.ndim == 1:
        return y[0]
    return y if not unflip else y.T

def filter_band(x: np.ndarray, f_band: tuple[float, float], fs: float) -> np.ndarray:
    """
    2nd-order Butterworth bandpass with zero-phase (filtfilt), like MATLAB:
    [b,a] = butter(2, f/(fs/2), 'bandpass'); y = filtfilt(b,a,x')'
    Works on (samples,) or (channels, samples).
    """
    nyq = 0.5 * fs
    Wn = (f_band[0] / nyq, f_band[1] / nyq)
    b, a = butter(2, Wn, btype="bandpass")
    x = np.asarray(x)
    if x.ndim == 1:
        return filtfilt(b, a, x)
    # ensure shape (channels, samples)
    X = x if x.shape[0] < x.shape[1] else x.T
    Y = np.vstack([filtfilt(b, a, ch) for ch in X])
    return Y if x.shape[0] < x.shape[1] else Y.T

def filter_harmonics(x: np.ndarray, f: float, fs: float, bw: float) -> np.ndarray:
    """
    Apply 2nd-order bandstop filters around every multiple of f up to fs/2 (excluding fs/2).
    In MATLAB: butter(2, harm * [1-bw, 1+bw] / (fs/2), 'stop'); filtfilt(...)
    bw is relative width (e.g., 0.01 for +/-1%).
    """
    x = np.asarray(x)
    # normalize to (channels, samples)
    if x.ndim == 1:
        X = x[None, :]
        squeeze = True
    else:
        X = x if x.shape[0] < x.shape[1] else x.T
        squeeze = False

    nyq = 0.5 * fs
    harmonics = [h for h in np.arange(f, nyq, f) if not np.isclose(h, nyq)]
    Y = X.astype(np.float64, copy=True)
    for h in harmonics:
        Wn = (h * (1 - bw) / nyq, h * (1 + bw) / nyq)
        b, a = butter(2, Wn, btype="bandstop")
        for i in range(Y.shape[0]):
            Y[i] = filtfilt(b, a, Y[i])

    if squeeze:
        return Y[0]
    return Y if x.shape[0] < x.shape[1] else Y.T
