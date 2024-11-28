import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as sig


def sine_signal(f: int, sample_rate: int, duration: int, A: float, phi: float) -> (np.ndarray, np.ndarray):
    t = np.linspace(0, duration, duration * sample_rate, endpoint=False)
    sinusoid = A * np.sin(2 * np.pi * f * t + phi)
    return sinusoid, t


def ex1(N: int) -> None:
    x: np.ndarray = np.random.rand(N)

    fig, axs = plt.subplots(4)

    axs[0].plot(x)

    for i in range(3):
        plot_number = i + 1
        reversed = np.flip(x, axis=0)
        result = []
        for j in range(N):
            sum = 0
            for k in range(j):
                sum += reversed[k] * x[k]
            result.append(sum)

        for j in range(N, -1, -1):
            sum = 0
            for k in range(j):
                sum += reversed[k] * x[k]

            result.append(sum)

        dim = N * 2
        x = np.array(result[dim // 4: (dim - dim // 4)])
        x = x / np.mean(x)
        axs[plot_number].plot(x)

    plt.tight_layout()
    plt.savefig('plots/ex1.png', bbox_inches='tight')
    plt.savefig('plots/ex1.pdf', bbox_inches='tight')
    plt.plot(np.convolve(x, x))
    plt.savefig('plots/ex1_2.png', bbox_inches='tight')
    plt.savefig('plots/ex1_2.pdf', bbox_inches='tight')


def complete_with_zeros(p: np.ndarray, q: np.ndarray) -> (np.ndarray, np.ndarray):
    if len(p) > len(q):
        q = np.pad(q, (0, len(p) - len(q)))
    else:
        p = np.pad(p, (0, len(q) - len(p)))

    return p, q


def compare_conv_fft(N: int) -> None:
    degree_of_p = np.random.randint(0, N)
    degree_of_q = np.random.randint(0, N)

    p = np.random.randint(-19030, 19030, size=degree_of_p)
    q = np.random.randint(-19030, 19030, size=degree_of_q)

    r = np.convolve(p, q)

    p, q = complete_with_zeros(p, q)

    p_fft = np.fft.fft(p, n=len(r))
    q_fft = np.fft.fft(q, n=len(r))

    r_fft = np.fft.ifft(p_fft * q_fft)

    print(np.linalg.norm(r - r_fft))


def rectangular_window(N: int) -> np.array:
    window = N * [1]
    return np.array(window)


def hanning_window(N: int) -> np.array:
    n = np.arange(0, N, 1)
    return .5 * (1 - np.cos(2 * np.pi * n / N))


def ex3(f: int, A: float, phi: float, sample_rate: int, duration: float) -> None:
    sinusoid, t = sine_signal(f, sample_rate, int(duration * sample_rate), A, phi)

    Nw = 200

    rectang_window = rectangular_window(Nw)
    hann_window = hanning_window(Nw)

    sinusoid_rectang = sinusoid[:Nw] * rectang_window

    sinusoid_hann = sinusoid[:Nw] * hann_window

    plt.figure(figsize=(12, 6))

    plt.subplot(3, 1, 1)
    plt.plot(t[:Nw], sinusoid[:Nw], label="Sinusoida originala")
    plt.xlabel("Timp")
    plt.ylabel("Amplitudine")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(t[:Nw], sinusoid_rectang, label="Sinusoida cu fereastra 1", color="orange")
    plt.xlabel("Timp")
    plt.ylabel("Amplitudine")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(t[:Nw], sinusoid_hann, label="Sinusoida cu fereastra 2", color="green")
    plt.xlabel("Timp")
    plt.ylabel("Amplitudine")
    plt.legend()

    plt.tight_layout()
    plt.savefig('plots/ex3.png', bbox_inches='tight')
    plt.savefig('plots/ex3.pdf', bbox_inches='tight')


def ex4_a(days: int):
    data = np.genfromtxt('csv_file/Train.csv', dtype=[int, 'U19', int], delimiter=',', skip_header=1)
    sel_data = [elem[2] for elem in data[: 24 * days]]
    return sel_data


def ex4_b(data: pd.Series, dim: list[int]) -> None:
    fig, axs = plt.subplots(len(dim))
    for index in range(len(dim)):
        plt.title(f"Size = {dim[index]}")
        filtered_signal = np.convolve(data, np.ones(dim[index]), 'valid')
        axs[index].plot(filtered_signal)
    plt.savefig('plots/ex4.png', bbox_inches='tight')
    plt.savefig('plots/ex4.pdf', bbox_inches='tight')



def main():
    ex1(100)
    compare_conv_fft(40)
    ex3(100, 1, 0, 1000, 0.2)

    data = ex4_a(3)
    data = np.array(data, dtype=float)
    ex4_b(data, [5, 9, 13, 17])

main()
