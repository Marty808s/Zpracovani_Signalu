import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt


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


def identify_words(signal, threshold, rules):
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


def visualize_signal(signal, amplitude_envelope, words):
    plt.plot(signal, label='Signál')
    plt.plot(amplitude_envelope, label='Amplitudový obal')
    for start, end in words:
        plt.axvspan(start, end, color='gray', alpha=0.5)
    plt.legend()
    plt.show()


# Načtení signálu ze souboru
signal_path = 'InputData/Signal1.txt'
signal = load_signal(signal_path)

# Identifikace slov v signálu s prahovou hodnotou 0.5 pomocí Hammingova okna
threshold = 0.5
words = identify_words(signal, threshold, None)

# Vizualizace signálu s identifikovanými slovy
visualize_signal(signal, apply_hilbert(apply_hamming(signal)), words)

# Zde můžete dále pracovat s identifikovanými slovy podle potřeby
print("Identifikovaná slova:", words)
