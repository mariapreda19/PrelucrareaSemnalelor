import matplotlib.pyplot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter

def calculate_freq(data: pd.DataFrame) -> float:
    timestamp1 = pd.to_datetime(data.iloc[0, 1], format='%d-%m-%Y %H:%M')
    timestamp2 = pd.to_datetime(data.iloc[1, 1], format='%d-%m-%Y %H:%M')

    time_diff = (timestamp2 - timestamp1).total_seconds()

    return time_diff

def total_time(data: pd.DataFrame) -> float:
    timestamp1 = pd.to_datetime(data.iloc[0, 1], format='%d-%m-%Y %H:%M')
    timestamp2 = pd.to_datetime(data.iloc[len(data) - 1, 1], format='%d-%m-%Y %H:%M')

    time_diff = (timestamp2 - timestamp1).total_seconds()

    return time_diff

def plot_fourier(data: pd.DataFrame) -> None:
    values = data.iloc[:, 2]
    fourier = np.fft.fft(values)[0:len(values) // 2]
    f =  np.linspace(0, len(values)//2, len(values)//2) / len(values) * (1/calculate_freq(data))

    plt.plot(f, np.abs(fourier))

    plt.savefig('plots/fourier.png')
    plt.savefig('plots/fourier.pdf')

def eliminate_cont_comp(data: pd.DataFrame) -> pd.DataFrame:
    values = data.iloc[:, 2]
    mean_val = np.mean(values)

    if mean_val != 0:
        values -= mean_val
        print("Componenta a fost eliminata")
    else:
        print("Nu exista")

    data.iloc[:, 2] = values
    return data


def calculate_main_frequencies(data: pd.DataFrame) -> pd.DataFrame:
    values = data.iloc[:, 2]
    fourier_transform = np.fft.fft(values)
    magnitude_spectrum = np.abs(fourier_transform)[:len(values) // 2]
    frequencies = np.fft.fftfreq(len(values), d=calculate_freq(data))[:len(values) // 2]

    top_indices = []
    magnitude_copy = magnitude_spectrum.copy()
    for _ in range(4):
        max_index = np.argmax(magnitude_copy)
        top_indices.append(max_index)
        magnitude_copy[max_index] = -1

    top_frequencies = [frequencies[i] for i in top_indices]
    top_magnitudes = [magnitude_spectrum[i] for i in top_indices]

    print("Primele 4 frecvențe principale sunt:", top_frequencies)
    print("Magnitudinile corespunzătoare sunt:", top_magnitudes)

    plt.figure(figsize=(10, 5))
    plt.plot(frequencies, magnitude_spectrum, label='Original Fourier Spectrum', zorder=1)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('Fourier Spectrum with Top 4 Frequencies Highlighted')

    for i in range(4):
        plt.scatter(top_frequencies[i], top_magnitudes[i], color='red', marker='o', s=50, alpha=0.7,
                    label=f'Top {i + 1} Frequency', zorder=2)

    plt.legend()
    plt.show()

    results = pd.DataFrame({
        "Frecvență (Hz)": top_frequencies,
        "Magnitudine": top_magnitudes
    })
    return results


def main():
    data = pd.read_csv('csvFiles/Train.csv', header=0)

    """
        Punctul a) calcularea frecventei de esantionare
    """

    f = calculate_freq(data)

    print('1/', f, 'Hz')


    """
        Punctul b) intervalul de timp ocupar de esantioane
    """

    print("Semnalul a fost inregistrat timp de ", total_time(data), " secunde")


    """
        Punctul c) frecventa maxima prezenta in semnal
    """

    print('Frecventa maxima este de 1 / (', 2 * f,') Hz')


    """
        Punctul d) Afisarea modulului transformatei Fourier
    """
    plot_fourier(data)

    """
        Punctul e) Eliminarea componentei continue
    """

    data = eliminate_cont_comp(data)

    """
       Punctul f
    """

    calculate_main_frequencies(data)

    """
        Punctul g
    """
    values = np.array(data.iloc[:, 2])
    plt.plot(values[1903:3019])
    plt.savefig('plots/signal.png')


    """
        Punctul h
    """

    cutoff = 0.07
    number_of_coef: int = 1903

    coef_filter = firwin(number_of_coef, cutoff)

    filtered_signal = lfilter(coef_filter, 1.0, values)

    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(values)
    plt.title('Original signal')

    plt.subplot(2, 1, 2)
    plt.plot(filtered_signal)
    plt.title('Filtered signal')

    plt.show()



main()

