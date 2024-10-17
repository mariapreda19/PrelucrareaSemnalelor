import numpy as np
import matplotlib.pyplot as plt
import math
import random
import sounddevice
from scipy.io import wavfile

def cosine_signal(f: int, t: np.ndarray, phase: float) -> np.ndarray:
    return np.cos(2 * np.pi * f * t + phase)

def sine_signal(f: int, t: np.ndarray, phase: float) -> np.ndarray:
    return np.sin(2 * np.pi * f * t + phase)


def sine_cosine_identical_plot(f: int, t: np.ndarray, phase:float) -> None:
    fig, axs = plt.subplots(2)
    fig.suptitle(f"Exercitiul 1")
    axs[0].plot(t, cosine_signal(f, t, phase))
    axs[1].plot(t, sine_signal(f, t, phase+np.pi/2))
    for ax in axs:
        ax.set(xlabel='time (s)', ylabel='Amplitude')
    file_path = 'plots/exercitiul1.png'
    plt.savefig(file_path, bbox_inches='tight')


def plot_different_phases(f: int, t: np.ndarray, number_of_phases: int) -> None:
    # all in the same plot
    sine_function_1 = sine_signal(f, t, 0)
    sine_function_2 = sine_signal(f, t, np.pi/2)
    sine_function_3 = sine_signal(f, t, np.pi)
    sine_function_4 = sine_signal(f, t, 3*np.pi/2)

    fig, ax = plt.subplots(4)
    fig.suptitle(f"Exercitiul 2")


    mean, standard_deviation = 0, 1
    z = np.random.normal(mean, standard_deviation, 100)

    def noise(SNR):
        return np.linalg.norm(2 * np.pi * f * t) / (np.sqrt(SNR) * np.linalg.norm(z))

    noise1 = noise(0.1)
    noise2 = noise(1)
    noise3 = noise(10)
    noise4 = noise(100)

    ax[0].plot(t, sine_function_1 + noise1 * z)
    ax[1].plot(t, sine_function_2 + noise2 * z)
    ax[2].plot(t, sine_function_3 + noise3 * z)
    ax[3].plot(t, sine_function_4 + noise4 * z)

    file_path = 'plots/exercitiul2.png'
    plt.savefig(file_path, bbox_inches='tight')


def sinusoidal_sawtooth_addition() -> None:
    # sinusoidal signal
    f = 100
    t = np.linspace(0, 0.1, 200)
    sinusoidal_signal = sine_signal(f, t, 0)

    # sawtooth signal
    sawtooth_signal = 2 * (f*t - np.floor(f*t))

    # addition of the two signals
    addition = sinusoidal_signal + sawtooth_signal

    fig, ax = plt.subplots(3)
    fig.suptitle(f"Exercitiul 3")

    ax[0].plot(t, sinusoidal_signal)
    ax[1].plot(t, sawtooth_signal)
    ax[2].plot(t, addition)

    file_path = 'plots/exercitiul3.png'
    plt.savefig(file_path, bbox_inches='tight')


def signal_audio(f: int, t: np.ndarray, phase: float, sample_rate: int) -> None:
    sounddevice.play(sine_signal(f, t, phase), samplerate=sample_rate)
    sounddevice.wait()


def signals_audio_lab1() -> None:
    sample_rate = 44100
    time = np.linspace(0, 10, sample_rate)

    signal_audio(400, time, 0, sample_rate)

    wavfile.write('audio/sine_wave_400Hz.wav', sample_rate, sine_signal(400, time, 0).astype(np.float32))

    frequency = 800
    time = np.arange(0, 3, 1 / (3 * frequency))
    function_2_b = np.sin(2 * np.pi * time * frequency)

    sounddevice.play(function_2_b, samplerate=sample_rate)
    sounddevice.wait()

    wavfile.write('audio/sine_wave_800Hz.wav', sample_rate, function_2_b.astype(np.float32))

    frequency = 240
    time = np.linspace(0, 0.1, 1000)

    function_2_c = 2 * (frequency * time - np.floor(frequency * time + 0.5))

    sounddevice.play(function_2_c, samplerate=sample_rate)
    sounddevice.wait()

    wavfile.write('audio/sawtooth_wave_240Hz.wav', sample_rate, function_2_c.astype(np.float32))

    frequency = 300
    time = np.linspace(0, 0.1, 1000)
    function_2_d = np.sign(np.sin(2 * np.pi * frequency * time))

    sounddevice.play(function_2_d, samplerate=sample_rate)
    sounddevice.wait()

    wavfile.write('audio/square_wave_300Hz.wav', sample_rate, function_2_d.astype(np.float32))





def signal_audio_combined() -> None:
    sample_rate = 44100
    duration = 10
    time = np.linspace(0, duration, duration * sample_rate)

    signal1 = sine_signal(400, time, 0)
    signal2 = sine_signal(600, time, 0)

    combined_signal = np.concatenate((signal1, signal2))

    sounddevice.play(combined_signal, samplerate=sample_rate)
    sounddevice.wait()

    wavfile.write('audio/combined_sine_wave_400Hz_600Hz.wav', sample_rate, combined_signal.astype(np.float32))



def ex6(fs: int, t: np.ndarray) -> None:
    f1 = fs/2
    f2 = fs/4
    f3 = 0

    signal1 = sine_signal(f1, t, 0)
    signal2 = sine_signal(f2, t, 0)
    signal3 = sine_signal(f3, t, 0)

    fig, ax = plt.subplots(3)
    fig.suptitle(f"Exercitiul 6")

    ax[0].plot(t, signal1)
    ax[1].plot(t, signal2)
    ax[2].plot(t, signal3)

    file_path = 'plots/exercitiul6.png'
    plt.savefig(file_path, bbox_inches='tight')


def ex7(f: int) -> None:
    fig, axs = plt.subplots(3)
    fig.suptitle("Signal decimation")

    time = np.linspace(0, 10, 1000)

    time_first_dec = time[::4] / 4
    time_second_dec = time_first_dec[1::4] / 4

    axs[0].plot(time, sine_signal(f, time, 0))
    axs[1].plot(time_first_dec, sine_signal(f, time_first_dec, 0))
    axs[2].plot(time_second_dec, sine_signal(f, time_second_dec, 0))

    file_path = 'plots/exercitiul7.png'
    plt.savefig(file_path, bbox_inches='tight')


def ex8() -> None:
    alpha = np.linspace(-np.pi / 2, np.pi / 2, 1000)

    sin_vals = np.sin(alpha)

    # Pade approximation for sin(alpha)
    pade = (alpha - (7 * alpha ** 3) / 60) / (1 + alpha ** 2 / 20)

    fig, axs = plt.subplots(2, figsize=(8, 10))

    # sin(x), x and pade approximation
    axs[0].plot(alpha, alpha, label='f(x) = x', linestyle='--', color='orange')
    axs[0].plot(alpha, sin_vals, label='sin(x)', color='blue')
    axs[0].plot(alpha, pade, label='Pade Approximation', color='green')

    axs[0].set_title('Comparison of sin(x), f(x) = x and Pade Approximation')
    axs[0].set_xlabel('Alpha (radians)')
    axs[0].set_ylabel('Amplitude')
    axs[0].legend(loc='best')
    axs[0].grid(True)

    # error
    axs[1].semilogy(alpha, np.abs(sin_vals - pade), label='|sin(x) - Pade|', color='green')
    axs[1].semilogy(alpha, np.abs(sin_vals - alpha), label='|sin(x) - x|', color='orange')

    axs[1].set_title('Error comparison (Log scale)')
    axs[1].set_xlabel('Alpha (radians)')
    axs[1].set_ylabel('Error (log scale)')
    axs[1].legend(loc='best')
    axs[1].grid(True)

    file_path = 'plots/exercitiul8.png'
    plt.savefig(file_path, bbox_inches='tight')



def __main__():
    sine_cosine_identical_plot(520, np.arange(0, 0.03, 0.0001), 0)
    plot_different_phases(1, np.linspace(0, 1, 100), 4)
    sinusoidal_sawtooth_addition()
    ex6(200, np.linspace(0, 1, 200))
    ex7(400)
    ex8()
    signals_audio_lab1()
    signal_audio_combined()


__main__()