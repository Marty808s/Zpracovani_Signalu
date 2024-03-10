import numpy as np
import wave
import scipy as sp
import matplotlib.pyplot as plt
import sympy as smp
import scipy.signal as sig
from scipy.integrate import quad
import wave, struct

t, f = smp.symbols('t, f', real=True)
k = smp.symbols('k', real=True, positive=True)
x = smp.exp(-k * t ** 2) * k * t


def median_filter(signal, window_median):
    kernel = np.ones(window_median) / window_median
    smoothed_signal = convolution(signal, kernel)
    return smoothed_signal


def convolution(signal, kernel):
    # Výstupní signál bude mít délku len(signal) + len(kernel) - 1
    output_length = len(signal) + len(kernel) - 1
    output = np.zeros(output_length)
    return output


def apply_hamming(signal):
    hamming_window = sig.get_window('hamming', len(signal))
    signal_hamming = signal * hamming_window
    return signal_hamming


def apply_hilbert(signal):
    analytic_signal = sig.hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)
    return amplitude_envelope


def apply_fourier_transform(signal, f=f):
    spectrum = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(signal), d=1/f)
    return frequencies, spectrum


def visualize_signal(signal, amplitude_envelope, words):
    plt.plot(signal, label='Signál')
    plt.plot(amplitude_envelope, label='Amplitudový obal')
    plt.axhline(y=0.5, color='r', linestyle='-')
    for start, end in words:
        plt.axvspan(start, end, color='gray', alpha=0.5)
    plt.legend()
    plt.show()


def identify_words(signal, threshold):
    words = []

    # Aplikace Hammingova okna na signál
    signal_hamming = apply_hamming(signal)

    # Aplikace Hilbertovy transformace na signál pro získání analytickeho signálu
    amplitude_envelope = apply_hilbert(signal_hamming)

    # Detekce přechodů signálu přes zvolený prah
    crossings = np.where(amplitude_envelope > threshold)[0]

    # Určení začátku a konce každého slova na základě přechodů
    word_start = crossings[0]
    for i in range(1, len(crossings)):
        if crossings[i] - crossings[i-1] > 1:
            words.append((word_start, crossings[i-1]))
            word_start = crossings[i]

    # Přidání posledního slova
    words.append((word_start, crossings[-1]))

    return words


wav_file = 'Samples/best_zaznam.wav'

threshold = 2

wavefile = wave.open(wav_file, 'r')

length = wavefile.getnframes()

data = []
num_channels = wavefile.getnchannels()
sample_width = wavefile.getsampwidth()

for i in range(0, length):
    wavedata = wavefile.readframes(1)
    if wavedata:
        # Načtení hodnot pro každý kanál
        if sample_width == 1:  # Pokud je šířka vzorku 1 byte
            values = struct.unpack(f"{num_channels}B", wavedata)
        elif sample_width == 2:  # Pokud je šířka vzorku 2 byty
            values = struct.unpack(f"{num_channels}h", wavedata)
        else:
            raise ValueError("Unsupported sample width")

        # Přidání hodnot každého kanálu do seznamu
        for j in range(num_channels):
            data.append(values[j])
    else:
        break


plt.plot(data)
plt.show()

print(len(data))
signal = data

window_median = 5
signal = median_filter(signal, window_median)

print(signal[:50000])

plt.plot(signal)
plt.show()

words = identify_words(signal, threshold)
print(len(words))

visualize_signal(signal, apply_hilbert(apply_hamming(signal)), words)