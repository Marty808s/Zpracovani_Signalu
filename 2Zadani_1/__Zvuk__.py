import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

def identify_words(signal, threshold):
    words = []

    #Aplikace Hammingova okna na signál
    hamming_window = sig.get_window('hamming', len(signal))
    signal_hamming = signal * hamming_window

    #Aplikace Hilbertovy transformace na signál pro získání analytickeho signálu
    analytic_signal = sig.hilbert(signal_hamming)
    amplitude_envelope = np.abs(analytic_signal)

    # Detekce přechodů signálu přes zvolený prah
    crossings = np.where(amplitude_envelope > threshold)[0]

    #Určení začátku a konce každého slova na základě přechodů
    word_start = crossings[0]
    for i in range(1, len(crossings)):
        if crossings[i] - crossings[i-1] > 1:
            words.append((word_start, crossings[i-1]))
            word_start = crossings[i]

    #Přidání posledního slova
    words.append((word_start, crossings[-1]))

    return words

#Načtení signálu ze souboru
with open('InputData/Signal1.txt', 'r') as file:
    signal_values = [float(line.strip()) for line in file.readlines()]

#Převod listu na numpy array
signal = np.array(signal_values)

#Identifikace slov v signálu s prahovou hodnotou 0.5 pomocí Hammingova okna
words = identify_words(signal, threshold=0.5)

print(words)
print(len(words))

plt.plot(signal_values)
plt.show()