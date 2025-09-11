import numpy as np
import matplotlib.pyplot as plt
import time
# Set parameters for the signal
N = 2048 
T = 1.0 
f0 = 1 # Number of sample points
# Duration of the signal in seconds
# Fundamental frequency of the square wave in Hz
# List of harmonic numbers used to construct the square wave
harmonics = [1, 3, 5, 50, 100]
# Define the square wave function
def square_wave(t):
    return np.sign(np.sin(2.0 * np.pi * f0 * t))
# Fourier series approximation of the square wave
def square_wave_fourier(t, f0, N):
    result = np.zeros_like(t)
    for k in range(N):
# The Fourier series of a square wave contains only odd harmonics.
        n = 2 * k + 1
# Add harmonics to reconstruct the square wave.
        result += np.sin(2 * np.pi * n * f0 * t) / n
    return (4 / np.pi) * result
# Create the time vector
# np.linspace generates evenly spaced numbers over a specified interval.
# We use endpoint=False because the interval is periodic.
t = np.linspace(0.0, T, N, endpoint=False)
# Generate the original square wave
square = square_wave(t)
plt.figure(figsize=(12, 8))
# Plot the original square wave
plt.subplot(2, 3, 1)
plt.plot(t, square,'k', label="Square wave")
plt.title("Original Square Wave")
plt.ylim(-1.5, 1.5)
plt.grid(True)
plt.legend()
# Plot Fourier reconstructions under different number of harmonics
for i, Nh in enumerate(harmonics, start=2):
    plt.subplot(2, 3, i)
    y = square_wave_fourier(t, f0, Nh)
    plt.plot(t, y, label=f"N={Nh} harmonics")
    plt.plot(t, square,'k--', alpha=0.5, label="Square wave")
    plt.title(f"Fourier Approximation with N={Nh}")
    plt.ylim(-1.5, 1.5)
    plt.grid(True)
    plt.legend()


plt.tight_layout()
plt.show()


# 2. Apply the DFT and time the execution
def naive_dft(x):
    """
    Compute the Discrete Fourier Transform (DFT) of a 1D signal.
    This is a "naïve" implementation that directly follows the DFT formula,
    which has a time complexity of O(N^2).
Args:
    x (np.ndarray): The input signal, a 1D NumPy array.
    Returns:
    np.ndarray: The complex-valued DFT of the input signal.
"""
    N = len(x)
# Create an empty array of complex numbers to store the DFT results
    X = np.zeros(N, dtype=np.complex128)
# Iterate through each frequency bin (k)
    for k in range(N):
# For each frequency bin, sum the contributions from all input samples (n)
        for n in range(N):
# The core DFT formula: x[n] * e^(-2j * pi * k * n / N)
            angle = -2j * np.pi * k * n / N
            X[k] += x[n] * np.exp(angle)
    return X
# Construct a square wave using 50 harmonics
signal = square_wave_fourier(t, f0, 50)
# Time the naïve DFT implementation
start_time_naive = time.time()
dft_result = naive_dft(signal)
end_time_naive = time.time()
naive_duration = end_time_naive - start_time_naive
# Time NumPy's FFT implementation
start_time_fft = time.time()
fft_result = np.fft.fft(signal)
end_time_fft = time.time()
fft_duration = end_time_fft - start_time_fft
# 3. Print Timings and Verification
print("--- DFT/FFT Performance Comparison ---")
print(f"Naïve DFT Execution Time: {naive_duration:.6f} seconds")
print(f"NumPy FFT Execution Time: {fft_duration:.6f} seconds")
# It's possible for the FFT to be so fast that the duration is 0.0, so we handle that case.
if fft_duration > 0:
    print(f"FFT is approximately {naive_duration / fft_duration:.2f} times faster.")
else:
    print("FFT was too fast to measure a significant duration difference.")
# Check if our implementation is close to NumPy's result
# np.allclose is used for comparing floating-point arrays.
print(f"\nOur DFT implementation is close to NumPy's FFT: {np.allclose(dft_result, fft_result)}")

def naive_dft(x):
    """
    Compute the Discrete Fourier Transform (DFT) of a 1D signal.
    This is a "naïve" implementation that directly follows the DFT formula,
    which has a time complexity of O(N^2).
    Args:
    x (np.ndarray): The input signal, a 1D NumPy array.
    Returns:
    np.ndarray: The complex-valued DFT of the input signal.
    """
    N = len(x)
    # Create an empty array of complex numbers to store the DFT results
    X = np.zeros(N, dtype=np.complex128)
    # Iterate through each frequency bin (k)
    for k in range(N):
        # For each frequency bin, sum the contributions from all input samples (n)
        for n in range(N):
            # The core DFT formula: x[n] * e^(-2j * pi * k * n / N)
            angle = -2j * np.pi * k * n / N
            X[k] += x[n] * np.exp(angle)
    return X
# Construct a square wave using 50 harmonics
signal = square_wave_fourier(t, f0, 50)
# Time the naïve DFT implementation
start_time_naive = time.time()
dft_result = naive_dft(signal)
end_time_naive = time.time()
naive_duration = end_time_naive - start_time_naive
# Time NumPy's FFT implementation
start_time_fft = time.time()
fft_result = np.fft.fft(signal)
end_time_fft = time.time()
fft_duration = end_time_fft - start_time_fft
# 3. Print Timings and Verification
print("--- DFT/FFT Performance Comparison ---")
print(f"Naïve DFT Execution Time: {naive_duration:.6f} seconds")
print(f"NumPy FFT Execution Time: {fft_duration:.6f} seconds")
# It's possible for the FFT to be so fast that the duration is 0.0, so we handle that case.
if fft_duration > 0:
    print(f"FFT is approximately {naive_duration / fft_duration:.2f} times faster.")
else:
    print("FFT was too fast to measure a significant duration difference.")
# Check if our implementation is close to NumPy's result
# np.allclose is used for comparing floating-point arrays.
print(f"\nOur DFT implementation is close to NumPy's FFT: {np.allclose(dft_result, fft_result)}")   

# 4. Prepare for Plotting
# Generate the frequency axis for the plot.
# np.fft.fftfreq returns the DFT sample frequencies.
# We only need the first half of the frequencies (the positive ones) due to symmetry.
xf = np.fft.fftfreq(N, T / N)[:N // 2]
# We normalize the magnitude by N and multiply by 2 to get the correct amplitude.
magnitude = 2.0/N * np.abs(dft_result[0:N//2])

# 5. Visualize the Results
plt.style.use('seaborn-v0_8-darkgrid')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot the original time-domain signal
ax1.plot(t, signal, color='c')
ax1.set_title('Input Sine Wave Signal', fontsize=16)
ax1.set_xlabel('Time (s)', fontsize=12)
ax1.set_ylabel('Amplitude', fontsize=12)
ax1.set_xlim(0, 1.0) # Show a few cycles of the sine wave
ax1.grid(True)

# Plot the frequency-domain signal (magnitude of the DFT)
ax2.stem(xf, magnitude, basefmt=" ")
ax2.set_title(
    'Discrete Fourier Transform (Magnitude Spectrum)',
    fontsize=16
    )
ax2.set_xlabel('Frequency (Hz)', fontsize=12)
ax2.set_ylabel('Magnitude', fontsize=12)
ax2.set_xlim(0, 50) # Focus on the lower frequencies
ax2.grid(True)

# Add vertical lines for the first ten frequencies
for i in range(20):
    if i < len(xf) and i % 2 == 1: # Only plot odd harmonics
        ax2.axvline(x=xf[i], color='r', linestyle='--', alpha=0.7,
                    label = f'f{i}: {i}* f0 = {xf[i]:.1f} Hz' 
                    )
# Only show labels for first 3 frequencies to avoid cluttering
ax2.legend()
plt.tight_layout()
plt.show()

# ========= 追加代码：对比“构造用到的谐波” vs “FFT得到的谱线” =========

# 如果你说“50阶/100阶”是指“用了50/100个奇次项”，
# 那最高奇数阶分别是 2*K-1（K为项数），即 99 和 199
num_terms_list = [5, 20, 50, 100]     # 这里放“项数K”，你可改成 [50, 100]
odd_caps = [2*K - 1 for K in num_terms_list]   # 转成最高奇数阶

def square_wave_fourier_upto(t, f0, odd_cap):
    """用到 1,3,5,...,odd_cap（含）的所有奇次项来重建方波"""
    kmax = (odd_cap - 1) // 2  # odd_cap=5 -> k=0,1,2 -> n=1,3,5
    result = np.zeros_like(t, dtype=float)
    for k in range(kmax + 1):
        n = 2*k + 1
        result += np.sin(2*np.pi*n*f0*t) / n
    return (4/np.pi) * result

fs = N / T
freqs = np.fft.rfftfreq(N, d=1/fs)  # 非负频率（更方便一侧幅度）
def fft_mag_one_sided(x):
    X = np.fft.rfft(x)
    mag = (2.0 / N) * np.abs(X)  # 一侧幅度
    if N % 2 == 0:
        mag[-1] /= 2.0  # Nyquist 特例
    return mag

# 只关心“f0 的奇数倍”这些频点（因为理想方波只有奇次谐波）
def odd_indices_up_to(max_n):
    return [n for n in range(1, max_n+1) if n % 2 == 1]

all_results = []

plt.figure(figsize=(12, 10))
for i, (K, cap) in enumerate(zip(num_terms_list, odd_caps), 1):
    # 1) 用到K个奇次项来重建方波
    sig = square_wave_fourier_upto(t, f0, cap)

    # 2) 做FFT并取一侧幅度
    mag = fft_mag_one_sided(sig)

    # 3) 拿出我们“当初用于构造”的奇次频点：n = 1,3,5,...,cap
    theo_n = odd_indices_up_to(cap)
    # 对应的理论幅度（理想方波 ±1 的傅里叶系数）
    theo_amp = np.array([4/(np.pi*n) for n in theo_n], dtype=float)

    # 4) 在FFT幅度里取对应bin（因为f0=1Hz，分辨率=1Hz -> 频点正好对齐到整数bin）
    fft_at_theo = mag[np.array(theo_n, dtype=int)]

    # 5) 误差统计
    mae = np.mean(np.abs(fft_at_theo - theo_amp))
    maxe = np.max(np.abs(fft_at_theo - theo_amp))
    all_results.append((K, cap, mae, maxe))

    # 6) 画对比图（只画到 cap+10 看得清晰）
    plt.subplot(2, 2, i)
    right = min(cap + 10, len(freqs)-1)
    plt.stem(freqs[:right+1], mag[:right+1], basefmt=" ", label="FFT magnitude")
    plt.stem(np.array(theo_n, float), theo_amp, basefmt=" ", markerfmt='o', linefmt='-', label="Theory 4/(πn)")
    plt.title(f"Spectrum: use {K} odd terms (max odd n={cap})")
    plt.xlabel("Frequency (Hz)"); plt.ylabel("Amplitude"); plt.grid(True); plt.legend()

plt.tight_layout()
plt.show()

print("\nK(项数) | 最高奇数阶 |     MAE     |  Max Error")
for K, cap, mae, maxe in all_results:
    print(f"{K:7d} | {cap:9d} | {mae:10.6f} | {maxe:10.6f}")

# ========== 可选：解释性打印 ==========
print("\n说明：")
print("1) 理想方波的频谱只在奇数倍频点有线谱，幅度 = 4/(πn)。")
print("2) 我们用到的最高奇数阶 cap 越大，时域更接近方波；频域到 cap 为止有线谱，cap 以上≈0。")
print("3) 若仍有偏差，常见原因是幅度归一化或索引取法；本例 f0=1Hz 且分辨率=1Hz，泄漏基本没有。")
