import numpy as np
import matplotlib.pyplot as plt


def sine_signal(f: int, t: np.ndarray, phase: float) -> np.ndarray:
    return np.sin(2 * np.pi * f * t + phase)

def cosine_signal(f: int, t: np.ndarray, amplitude: float) -> np.ndarray:
    return np.cos(2 * np.pi * f * t) * amplitude

def create_fourier_matrix(n: int) -> np.ndarray:
    fourier_matrix = np.zeros((n, n), dtype=complex)
    for i in range(n):
        for ii in range(n):
            fourier_matrix[i][ii] = np.exp(-2 * np.pi * 1j * i * ii / n)
    return fourier_matrix


def plot_fourier_matrix(F: np.ndarray) -> None:
    N = F.shape[0]
    fig, axs = plt.subplots(N, 2, figsize=(10, 15))
    fig.suptitle("Fourier matrix")

    for i in range(N):
        axs[i, 0].plot(np.real(F[i]))
        axs[i, 0].set_title(f"Real part of row {i}")
        axs[i, 1].plot(np.imag(F[i]))
        axs[i, 1].set_title(f"Imaginary part of row {i}")

    plt.tight_layout()

    path = 'plots/fourier_matrix'
    plt.savefig(path + '.png', bbox_inches='tight')
    plt.savefig(path + '.pdf', bbox_inches='tight')


def check_unitarity(F: np.ndarray) -> bool:
    F_H = np.conjugate(F.T)
    identity_check = np.dot(F, F_H)
    return np.allclose(identity_check, np.eye(F.shape[0]) * F.shape[0])


def ex_2_fig_1(f: int, time: np.ndarray, phase: float) -> None:

    sine_function = sine_signal(f, time, phase)

    real_part = sine_function * np.cos(-2 * np.pi * time)
    imaginary_part = sine_function * np.sin(-2 * np.pi * time)

    point = 620

    fig, ax = plt.subplots(1, 2)
    ax[0].axhline(y=0)
    ax[0].plot(time, sine_function)

    ax[0].plot(time[point], sine_function[point], color='red')
    ax[0].plot(2*[time[point]], [0, sine_function[point]], color='red')

    ax[1].axhline(y=0, color='black', linewidth='1')
    ax[1].axvline(x=0, color='black', linewidth='1')

    ax[1].plot(real_part[point], imaginary_part[point], color='red')
    ax[1].plot([0, real_part[point]], [0, imaginary_part[point]], color='red', linewidth=1.2)


    ax[1].scatter(real_part, imaginary_part, s=1)
    ax[1].set_xlabel('Real')
    ax[1].set_ylabel('Imaginary')
    ax[1].set_aspect('equal', adjustable='box')

    plt.tight_layout()
    path = 'plots/exercitiul2'
    plt.savefig(path + '.png', bbox_inches='tight')
    plt.savefig(path + '.pdf', bbox_inches='tight')


def ex_2_fig_2(f: int, time: np.ndarray, phase: float) -> None:
    sine_function = sine_signal(f, time, phase)

    omega_freq = [3, 5, 7, 9]

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    for i, ax in enumerate(axs.flat):
        omega = omega_freq[i]
        real_part = sine_function * np.cos(-2 * np.pi * omega * time)
        imaginary_part = sine_function * np.sin(-2 * np.pi * omega * time)

        ax.axhline(y=0, color='black', linewidth=1)
        ax.axvline(x=0, color='black', linewidth=1)
        ax.set_xlabel('Real')
        ax.set_ylabel('Imaginary')
        ax.set_aspect('equal', adjustable='box')
        ax.title.set_text(f'ω = {omega}')

        ax.scatter(real_part, imaginary_part, s=1, label='Scatter')
        ax.plot(real_part, imaginary_part, color='red', label='Trajectory')

    plt.tight_layout()

    plt.savefig('plots/exercitiul2_2.png', bbox_inches='tight')
    plt.savefig('plots/exercitiul2_2.pdf', bbox_inches='tight')


def ex_3(freq: list[int], amplitude: list[float], time: np.ndarray) -> None:
    cosine_function_1 = cosine_signal(freq[0], time, amplitude[0])
    cosine_function_2 = cosine_signal(freq[1], time, amplitude[1])
    cosine_function_3 = cosine_signal(freq[2], time, amplitude[2])

    signal = cosine_function_1 + cosine_function_2 + cosine_function_3

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    axs[0].plot(time, signal)

    fundamental_freq = signal.shape[0] // 400

    X = np.empty((400, 1), dtype=complex)
    frequencies = np.array((range(400))) * fundamental_freq

    for i, fr in enumerate(frequencies):
        X[i] = np.sum(signal * np.exp(-2 * np.pi * 1j * fr * time))

    axs[1].stem(frequencies[:30], np.abs(X)[:30])

    plt.savefig('plots/exercitiul3.png', bbox_inches='tight')
    plt.savefig('plots/exercitiul3.pdf', bbox_inches='tight')


def __main__():

    """
        Exercitiul 1
    """
    n = 8
    F = create_fourier_matrix(n)
    plot_fourier_matrix(F)
    is_unitary = check_unitarity(F)

    if is_unitary:
        print("It's unitary")
    else:
        print("It's not unitary")

    """
        Exercitiul 2
    """

    ex_2_fig_1(7, np.linspace(0, 1, 1000), np.pi/2)

    ex_2_fig_2(7, np.linspace(0, 1, 1000), np.pi/2)


    """
        Exercitiul 3
    """

    ex_3([75, 5, 20], [3, 1.5, 3], np.linspace(0, 1, 2000))


__main__()

