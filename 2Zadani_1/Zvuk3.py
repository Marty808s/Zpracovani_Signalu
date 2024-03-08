import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sympy as smp
import scipy.signal as sig
from scipy.integrate import quad


t, f = smp.symbols('t, f', real=True)
k = smp.symbols('k', real=True, positive=True)
x = smp.exp(-k * t ** 2) * k * t

f = 22050

dict = ('time', 'prepare', 'solution', 'make', 'mistake', 'no', 'the', 'probable', 'long', 'lecture', 'method',
     'disaster', 'fail', 'work', 'advice', 'idea', 'succeed', 'easy', 'is', 'for', 'give')


def load_signal(file_path):
    with open(file_path, 'r') as file:
        signal_values = [float(line.strip()) for line in file.readlines()]
    return np.array(signal_values)


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


def sort_values(signal, indexes):
    vals = {}

    for num, i in enumerate(indexes):
        start_index, end_index = i
        sorted_val = signal[start_index:end_index]
        vals[num] = sorted_val

    return vals


def get_letters(dict=dict):
    letters_in = []
    for word in dict:
        for i in word:
            if i not in letters_in:
                letters_in.append(i)
            else:
                continue
    return letters_in


# Načtení signálu ze souboru
signal_path = 'InputData/Signal1.txt'
signal = load_signal(signal_path)
threshold = 0.5

words = identify_words(signal,threshold)

# Vizualizace signálu s identifikovanými slovy
visualize_signal(signal, apply_hilbert(apply_hamming(signal)), words)

# Zde můžete dále pracovat s identifikovanými slovy podle potřeby
print("Identifikovaná slova - indexy:", words, '\n', len(words))

sorted_words = sort_values(signal,words)

letters = get_letters()
print(letters,len(letters))

fourier = {}

for index, values in sorted_words.items():
    plt.plot(values, label=f"Values {index}")
    plt.show()
    freq, spect = apply_fourier_transform(values, f)
    fourier[index] = freq,spect
    plt.plot(freq, np.abs(spect))
    plt.show()

print(fourier)


