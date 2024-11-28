import numpy as np
import matplotlib.pyplot as plt
import time


def generate_time_series(N: int) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    times = np.arange(N)

    trend = 19 + 3 * times + times ** 2
    trend = trend / np.max(trend)

    seasonal = 3 * np.sin(2 * np.pi * 19 * times) + 19 * np.sin(2 * np.pi * 3 * times)

    residual = np.random.normal(0, 1, N)
    time_series = trend + seasonal + residual

    time_series = time_series / np.max(time_series)

    return time_series, trend, seasonal, residual


def plot_time_series(time_series: np.ndarray, trend: np.ndarray, seasonal: np.ndarray, residual: np.ndarray) -> None:
    fig, axs = plt.subplots(4)

    axs[0].set_title("Time series")
    axs[0].plot(time_series, color="orange")

    axs[1].set_title("Trend")
    axs[1].plot(trend)

    axs[2].set_title("seasonal ")
    axs[2].plot(seasonal)

    axs[3].set_title("residual")
    axs[3].plot(residual)

    plt.tight_layout()
    plt.savefig("plots/time_series.png")
    plt.savefig("plots/time_series_a.pdf")
    plt.clf()


def autocorrelation(time_series: np.ndarray) -> np.ndarray:
    autocor = [
        np.sum(time_series[:len(time_series) - i] * time_series[i:])
        for i in range(len(time_series))
    ]
    autocor = np.array(autocor)
    autocor /= max(autocor)

    return autocor


def plot_comparison_b(time_series: np.ndarray, autocor: np.ndarray) -> None:
    fig, axs = plt.subplots(2, figsize=(10, 6))

    axs[0].set_title("Autocorrelation")
    axs[0].plot(autocor, label="Autocorrelation", color="blue")
    axs[0].legend()

    predef_autocor = np.correlate(time_series, time_series, mode="full")
    predef_autocor = predef_autocor[len(predef_autocor) // 2:]
    predef_autocor /= predef_autocor[0]

    axs[1].set_title("Predefined Autocorrelation")
    axs[1].plot(predef_autocor, label="Predefined Autocorrelation", color="green")
    axs[1].legend()

    plt.tight_layout()
    plt.savefig("plots/autocorrelation.png")
    plt.savefig("plots/autocorrelation.pdf")
    plt.close()
def __main__():
    ts, trend, seasonal, residual = generate_time_series(1000)
    plot_time_series(ts, trend, seasonal, residual)
    plot_comparison_b(ts, autocorrelation(ts))



if __name__ == "__main__":
    __main__()