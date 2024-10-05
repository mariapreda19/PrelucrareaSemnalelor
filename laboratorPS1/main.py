import numpy as np
import matplotlib.pyplot as plt
import math
import random


"""
    Exercitiul 1
    
    x(t) = cos(520πt + π/3)
    y(t) = cos(280πt − π/3) 
    z(t) = cos(120πt + π/3)
    
"""

def calculate_signal(f: int, t: np.ndarray, phase: float) -> np.ndarray:
    return np.cos(2 * np.pi * f * t + phase)


t = 0.0005
time = np.arange(0, 0.03, t)

fig, axs = plt.subplots(3)
fig.suptitle("Exercitiul 1")
axs[0].plot(time, calculate_signal(520, time, np.pi/3))
axs[1].plot(time, calculate_signal(280, time, -np.pi/3))
axs[2].plot(time, calculate_signal(120, time, np.pi/3))



for ax in axs:
    ax.set(xlabel='time (s)', ylabel='Amplitude')

fig.show()


# pentru o reprezentare mai clara a semnalelor, putem scadea pasul de esantionare
t = 0.0001
time = np.arange(0, 0.03, t)

fig, axs = plt.subplots(3)
fig.suptitle("Exercitiul 1")
axs[0].plot(time, calculate_signal(520, time, np.pi/3))
axs[1].plot(time, calculate_signal(280, time, -np.pi/3))
axs[2].plot(time, calculate_signal(120, time, np.pi/3))



for ax in axs:
    ax.set(xlabel='time (s)', ylabel='Amplitude')

file_path = 'images/exercitiul1.png'
plt.savefig(file_path, bbox_inches='tight')

fig.show()


# Punctul c

frequency = 200

time = np.linspace(0, 0.03, 200)

fig, axs = plt.subplots(3)
fig.suptitle("Exercitiul 1, C")
axs[0].stem(time, calculate_signal(520, time, np.pi/3))
axs[1].stem(time, calculate_signal(280, time, -np.pi/3))
axs[2].stem(time, calculate_signal(120, time, np.pi/3))

axs[0].plot(time, calculate_signal(520, time, np.pi/3))
axs[1].plot(time, calculate_signal(280, time, -np.pi/3))
axs[2].plot(time, calculate_signal(120, time, np.pi/3))

for ax in axs:
    ax.set(xlabel='time (s)', ylabel='Amplitude')


file_path = 'images/exercitiul1c.png'
plt.savefig(file_path, bbox_inches='tight')
plt.show()


"""

    Exercitiul 2

"""


"""
    a) semnal sinusoidal cu frecventa 400Hz, 1600 esantioane
"""

frequency = 400
time = np.arange(0, 1, 1/1600)
function_2_a = np.sin(2 * np.pi * time * frequency)

plt.title("Exercitiul 2, a")
plt.plot(time[:math.floor(0.1 * len(time))], function_2_a[:math.floor(0.1 * len(time))])
plt.xlabel("time (s)")
plt.ylabel("Amplitude")


file_path = 'images/exercitiul2a.png'
plt.savefig(file_path, bbox_inches='tight')
plt.show()

"""
    b) Un semnal sinusoidal de frecventa 800 Hz, care sa dureze 3 secunde 
"""

frequency = 800
time = np.arange(0, 3, 1 / (3 * frequency))
function_2_b = np.sin(2 * np.pi * time * frequency)

plt.title("Exercitiul 2, b")
plt.plot(time[:math.floor(0.01 * len(time))], function_2_b[:math.floor(0.01 * len(time))])
plt.xlabel("time (s)")
plt.ylabel("Amplitude")


file_path = 'images/exercitiul2b.png'
plt.savefig(file_path, bbox_inches='tight')

plt.show()
"""
    c) Un semnal de tip sawtooth cu frecventa 240Hz
"""


frequency = 240
time = np.linspace(0, 0.1, 1000)

function_2_c = 2 * (frequency*time - np.floor(frequency*time + 0.5))


plt.plot(time, function_2_c)
plt.title('Exercitiul 2, c')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)

file_path = 'images/exercitiul2c.png'
plt.savefig(file_path, bbox_inches='tight')

plt.show()

"""
    d) un semnal de tip square cu frecventa 300Hz
"""

frequency = 300
time = np.linspace(0, 0.1, 1000)
function_2_d = np.sign(np.sin(2 * np.pi * frequency * time))

plt.title("Exercitiul 2, D")
plt.plot(time, function_2_d)
plt.xlabel("time (s)")
plt.ylabel("Amplitude")


file_path = 'images/exercitiul2d.png'
plt.savefig(file_path, bbox_inches='tight')
plt.show()

"""
    e) Un semnal 2D aleator. Creati un numpy.array de dimensiune 128x128 si initializati-l aleator.
    
"""

random_signal = np.random.rand(128, 128)
plt.title("Exercitiul 2, E")
plt.imshow(random_signal, cmap='gray')

file_path = 'images/exercitiul2e.png'
plt.savefig(file_path, bbox_inches='tight')
plt.show()

"""
    f) Un semnal 2D la alegerea voastra. Creati un numpy.array de dimensiune 128x128 si initializati-l
     folosind o procedura creata de voi. 

"""

def generate_signal() -> np.ndarray:
    signal_mat = np.zeros((128, 128))
    np.subtract(signal_mat, np.random.rand(128, 128))
    for i in range(128):
        for j in range(128):
            signal_mat[i][j] = signal_mat[i][j] +  random.random() * np.cos(2 * np.pi * (i + j))

    return signal_mat


file_path = 'images/exercitiul2f.png'
plt.imsave(file_path, generate_signal(), cmap='gray')
plt.show()

"""
    Exercitiul 3:
    
    Un semnal cu o frecventa de esantionare de 2000Hz
    
    a) Care este intervalul de timp intre doua esantioane? R: 1/2000
    
    b) Daca un esantion este memorat pe 4 biti, cati bytes vor ocupa 1 ora de achizitie?
    
    3600 secunde => numarul de esantioane este 3600 * 2000 = 72 * 10^5
    
    Un esantion va ocupa 0.5 bytes => 72 * 10^5 * 0.5 = 36 * 10^5 bytes
    
"""













