import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
from laborator3.laborator3 import sine_signal, create_fourier_matrix

def fft_vs_dft(size_of_matrix: list[int]) -> (list[float], list[float]):
    time_dft = []
    time_fft = []

    for n in size_of_matrix:
        t = np.linspace(0, 1, n)
        sin = sine_signal(1, t, 0)
        t_init = time.time()
        mat1 = np.matmul(create_fourier_matrix(n), sin)
        t_fin = time.time()
        time_dft.append(t_fin - t_init)

        t_init = time.time()
        mat2 = np.fft.fft(sin)
        t_fin = time.time()
        time_fft.append(t_fin - t_init)
        with open('fft_vs_dft.pickle', 'wb') as f:
            pickle.dump((mat1, mat2), f)

    return time_dft, time_fft

def plot_fft_vs_dft(time_dft: list[float], time_fft: list[float], size_of_matrix: list[int]) -> None:
    plt.yscale('log')
    plt.stem(size_of_matrix, time_dft, label='DFT')
    plt.stem(size_of_matrix, time_fft, label='FFT', markerfmt='C1x')
    plt.legend()
    plt.xlabel("Matrix Size")
    plt.ylabel("Time (log scale)")
    plt.title("DFT vs FFT Time Comparison")
    plt.show()


def generate_sampled_signals_ex2(f0: int, fs: int, A: float, phi: float, t: np.ndarray) ->\
        (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    ts = np.linspace(0, 1, fs)
    samples = A * np.sin(2 * np.pi * f0 * ts + phi)

    sin1 = A * np.sin(2 * np.pi * f0 * t + phi)
    sin2 = A * np.sin(2 * np.pi * (f0 - 1 * (fs - 1)) * t + phi)
    sin3 = A * np.sin(2 * np.pi * (f0 - 2 * (fs - 1)) * t + phi)

    return sin1, sin2, sin3, samples, ts


def plot_sampled_signals_ex2(t: np.ndarray, signals: list, samples: np.ndarray, ts: np.ndarray) -> None:
    fig, axs = plt.subplots(4, figsize=(10, 10))
    sin1, sin2, sin3 = signals[:3]

    for i, ax in enumerate(axs):
        if i == 0:
            ax.plot(t, sin1)
            ax.set_title("Sinusoidal Signal 1")
            continue
        ax.plot(ts, samples, marker='o', linestyle='None', markersize=7)
        ax.plot(t, signals[i - 1])
        ax.set_title(f"Sampled Signal {i}")

    plt.tight_layout()
    plt.show()

def generate_sampled_signals_ex3(f0: int, fs: int, A: float, phi: float, t: np.ndarray, new_fs: int) -> \
        (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):

    ts = np.linspace(0, 1, new_fs)
    samples = A * np.sin(2 * np.pi * f0 * ts + phi)

    sin1 = A * np.sin(2 * np.pi * f0 * t + phi)
    sin2 = A * np.sin(2 * np.pi * (f0 - 1 * (fs - 1)) * t + phi)
    sin3 = A * np.sin(2 * np.pi * (f0 - 2 * (fs - 1)) * t + phi)

    return sin1, sin2, sin3, samples, ts

def plot_sampled_signals_ex3(t: np.ndarray, signals: list, samples: np.ndarray, ts: np.ndarray) -> None:
    fig, axs = plt.subplots(4, figsize=(10, 10))
    sin1, sin2, sin3 = signals[:3]

    for i, ax in enumerate(axs):
        if i == 0:
            ax.plot(t, sin1)
            ax.set_title("Sinusoidal Signal 1")
            continue
        ax.plot(ts, samples, marker='o', linestyle='None', markersize=7)
        ax.plot(t, signals[i - 1])
        ax.set_title(f"Sampled Signal {i}")

    plt.tight_layout()
    plt.show()

def main_lab4() -> None:
    size_of_matrix = [128, 256, 512, 1024, 2048, 4096, 8192]

    # Part 1: DFT vs FFT timing
    try:
        with open('fft_vs_dft.pickle', 'rb') as f:
            pass
    except (FileNotFoundError, EOFError):
        time_dft, time_fft = fft_vs_dft(size_of_matrix)
        with open('fft_vs_dft.pickle', 'wb') as f:
            pickle.dump((time_dft, time_fft), f)
            plot_fft_vs_dft(time_dft, time_fft, size_of_matrix)




    sin1, sin2, sin3, samples, ts = generate_sampled_signals_ex2(10, 7, 1, 0, np.linspace(0, 1, 1000))
    plot_sampled_signals_ex2(np.linspace(0, 1, 1000), [sin1, sin2, sin3], samples, ts)

    t = np.linspace(0, 1, 1000)
    f0, fs, A, phi, new_fs = 10, 7, 1, 0, 20
    sin1, sin2, sin3, samples, ts = generate_sampled_signals_ex3(f0, fs, A, phi, t, new_fs)
    plot_sampled_signals_ex3(t, [sin1, sin2, sin3], samples, ts)

main_lab4()
