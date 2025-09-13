#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fourier (re)synthesis + DFT timing with NumPy vs PyTorch (CPU / CUDA / MPS)
----------------------------------------------------------------------------
This script fulfills the assignment requirements:

1) Re-implement square_wave, square_wave_fourier, and naive_dft using **PyTorch**
   tensor operations (no torch.fft for the naive DFT).
2) Provide a second naive DFT version that **explicitly runs on the accelerator**
   (CUDA on NVIDIA GPUs or MPS on Apple Silicon) using PyTorch tensors.
3) Compare computation times with NumPy's FFT (baseline) and NumPy's naive DFT.
4) Vary the size N and report the ranking from fastest to slowest for each N.
5) Explain observed ordering: FFT (O(N log N)) vs naive DFT (O(N^2)); accelerator
   parallelism reduces constants but does not change complexity.

Usage
-----
    pip install numpy matplotlib torch
    python fourier_pytorch_benchmark_universal.py

Notes
-----
- Accelerator selection is automatic: CUDA > MPS > CPU.
- If no accelerator is available, the accelerator path prints N/A.
- Beware of memory for naive DFT (builds an NxN complex matrix). Keep N moderate.
"""

import time
import math
import numpy as np
import matplotlib.pyplot as plt

try:
    import torch
except Exception as e:
    torch = None
    print("[WARN] PyTorch not available. NumPy-only parts will run.\n", e)

# -----------------------------
# Timing helpers
# -----------------------------
def _now():
    return time.perf_counter()

def _sync_if_needed(device: str):
    # Only CUDA needs explicit synchronize for fair timing.
    if torch is not None and device == "cuda":
        torch.cuda.synchronize()

# -----------------------------
# Device selection: CUDA > MPS > CPU
# -----------------------------
if torch is not None:
    if torch.cuda.is_available():
        DEVICE_ACCEL = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        DEVICE_ACCEL = "mps"
    else:
        DEVICE_ACCEL = "cpu"
else:
    DEVICE_ACCEL = "cpu"

# -----------------------------
# NumPy reference implementations
# -----------------------------
def square_wave_numpy(t, f0=1.0):
    return np.sign(np.sin(2.0 * np.pi * f0 * t))

def square_wave_fourier_numpy(t, f0=1.0, terms=51):
    # x(t) = (4/pi) * sum_{n odd}^{terms} sin(2π n f0 t)/n
    odd_ns = np.arange(1, terms + 1, 2)
    res = np.zeros_like(t, dtype=np.float64)
    for n in odd_ns:
        res += np.sin(2 * np.pi * n * f0 * t) / n
    return (4 / np.pi) * res

def naive_dft_numpy(x):
    # O(N^2) dense DFT via matrix multiply
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape(N, 1)
    W = np.exp(-2j * np.pi * k * n / N)  # [N,N]
    return W @ x.astype(np.complex128)

# -----------------------------
# PyTorch implementations (no torch.fft used for naive DFT)
# -----------------------------
def square_wave_torch(t, f0=1.0):
    """
    t: 1-D torch tensor on any device (cpu/cuda/mps), dtype float32/float64
    returns: sign(sin(2*pi*f0*t)) on same device/dtype
    """
    return torch.sign(torch.sin(2.0 * math.pi * f0 * t))

def square_wave_fourier_torch(t, f0=1.0, terms=51):
    """
    Finite Fourier synthesis using odd harmonics up to `terms` (inclusive).
    """
    dtype = t.dtype
    device = t.device
    # odd indices: 1,3,5,...
    odd_ns = torch.arange(1, terms + 1, 2, device=device, dtype=dtype)  # [M]
    # Broadcast t [N] with odd_ns [M]
    s = torch.sin(2.0 * math.pi * f0 * t.view(-1, 1) * odd_ns) / odd_ns
    res = s.sum(dim=1)
    return (4.0 / math.pi) * res

def naive_dft_torch(x):
    """
    O(N^2) DFT using PyTorch tensor ops on the **current device of x**.
    No torch.fft is used. Works on cpu/cuda/mps depending on x.device.
    """
    device = x.device
    dtype_c = torch.complex64 if x.dtype == torch.float32 else torch.complex128
    N = x.shape[0]
    n = torch.arange(N, device=device, dtype=torch.float32)
    k = n.view(-1, 1)
    # Build dense DFT matrix
    W = torch.exp(-2j * math.pi * k @ n.view(1, -1) / N).to(dtype=dtype_c)  # [N,N]
    x_c = x.to(dtype=dtype_c)
    return W @ x_c

# -----------------------------
# One-sided spectrum helper (NumPy)
# -----------------------------
def one_sided_amplitude_from_fft_numpy(X_full):
    N = X_full.shape[0]
    X = X_full[: N // 2 + 1]  # rfft-like slice
    mag = (2.0 / N) * np.abs(X)
    if N % 2 == 0:
        mag[-1] /= 2.0
    return mag

# -----------------------------
# Benchmark runner
# -----------------------------
def run_benchmarks(
    N_list=(256, 512, 1024, 2048),
    f0=1.0,
    duration_seconds=1.0,
    fourier_terms=101,
    make_plot_for_last=False,
):
    print("\n=== Fourier/DFT Timing: NumPy vs PyTorch (CPU / CUDA / MPS) ===")
    print(f"PyTorch available: {torch is not None}")
    if torch is not None:
        print(f"Selected accelerator device: {DEVICE_ACCEL}")

    for N in N_list:
        fs = N / duration_seconds
        t_np = np.linspace(0.0, duration_seconds, N, endpoint=False)

        # Build signal via NumPy Fourier synthesis (reference content)
        x_np = square_wave_fourier_numpy(t_np, f0=f0, terms=fourier_terms)

        # ----- NumPy FFT timing -----
        t0 = _now()
        X_fft_np = np.fft.fft(x_np)
        t1 = _now()
        numpy_fft_time = t1 - t0

        # ----- NumPy naive DFT timing -----
        t0 = _now()
        X_dft_np = naive_dft_numpy(x_np)
        t1 = _now()
        numpy_dft_time = t1 - t0

        # ----- PyTorch naive DFT on CPU -----
        if torch is not None:
            x_t_cpu = torch.from_numpy(x_np).to("cpu")
            t0 = _now()
            X_dft_t_cpu = naive_dft_torch(x_t_cpu)
            _sync_if_needed("cpu")
            t1 = _now()
            torch_cpu_time = t1 - t0
        else:
            torch_cpu_time = float("nan")

        # ----- PyTorch naive DFT on Accelerator (CUDA or MPS) -----
        if torch is not None and DEVICE_ACCEL in ("cuda", "mps"):
            x_t_accel = torch.from_numpy(x_np).to(DEVICE_ACCEL)
            _sync_if_needed(DEVICE_ACCEL)
            t0 = _now()
            X_dft_t_accel = naive_dft_torch(x_t_accel)
            _sync_if_needed(DEVICE_ACCEL)
            t1 = _now()
            torch_accel_time = t1 - t0
        else:
            torch_accel_time = float("nan")

        # Report
        methods = [
            ("NumPy FFT (O(N log N))", numpy_fft_time),
            ("NumPy naive DFT (O(N^2))", numpy_dft_time),
            ("Torch naive DFT CPU (O(N^2))", torch_cpu_time),
            (f"Torch naive DFT {DEVICE_ACCEL.upper()} (O(N^2))", torch_accel_time),
        ]

        # sort, treating NaN as +inf
        def _key(x):
            name, tt = x
            return math.inf if (tt != tt) else tt  # NaN check via (tt!=tt)

        methods_sorted = sorted(methods, key=_key)

        print(f"\nN={N:5d}  (fs={fs:.1f} Hz, duration={duration_seconds:.2f} s, terms={fourier_terms})")
        for name, tt in methods:
            if tt != tt:  # NaN
                print(f"  {name:35s} : N/A")
            else:
                print(f"  {name:35s} : {tt*1e3:8.3f} ms")

        print("  ==> Fastest -> Slowest:")
        rank_str = "  >  ".join([f"{name}" for name, tt in methods_sorted if tt == tt])
        print("     " + rank_str)

        # Optional: plot spectrum vs theory for the last N
        if make_plot_for_last and N == N_list[-1]:
            freqs = np.fft.rfftfreq(N, d=1/fs)
            mag = one_sided_amplitude_from_fft_numpy(X_fft_np)
            odd_ns = np.arange(1, int((fs/2)//f0)+1, 2)
            theo = 4/(np.pi*odd_ns)
            plt.figure(figsize=(10,5))
            plt.stem(freqs, mag, basefmt=" ", use_line_collection=True, label="NumPy FFT magnitude (one-sided)")
            plt.scatter(odd_ns*f0, theo, label="Theory 4/(πn) at odd harmonics", s=35)
            plt.xlim(0, min(60, fs/2))
            plt.xlabel("Frequency (Hz)"); plt.ylabel("Amplitude")
            plt.title(f"Spectrum vs Theory (N={N})")
            plt.grid(True); plt.legend(); plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    # Try larger N cautiously; naive DFT is O(N^2) and builds an NxN matrix.
    run_benchmarks(
        N_list=(256, 512, 1024, 2048),  # you can test (4096,) if memory allows
        f0=1.0,
        duration_seconds=1.0,
        fourier_terms=101,
        make_plot_for_last=False
    )
