import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sympy as smp
import scipy.signal as sig
from scipy.integrate import quad
from scipy.interpolate import interp1d

# Definice symbolických proměnných (pokud nejsou dále použity, lze je odstranit)
t, f = smp.symbols('t, f', real=True)
k = smp.symbols('k', real=True, positive=True)
x = smp.exp(-k * t ** 2) * k * t

# Vzorkovací frekvence
f = 22050

# Slovník slov (poznámka: nepoužívejte 'dict' jako název proměnné)
word_list = ('time', 'prepare', 'solution', 'make', 'mistake', 'no', 'the', 'probable', 'long', 'lecture', 'method',
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
    plt.figure(figsize=(12, 6))
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
    # Aplikace Hilbertovy transformace na signál pro získání analytického signálu
    amplitude_envelope = apply_hilbert(signal_hamming)
    # Detekce přechodů signálu přes zvolený práh
    crossings = np.where(amplitude_envelope > threshold)[0]
    if len(crossings) == 0:
        return words  # Žádná slova nebyla detekována
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

def get_letters(word_list=word_list):
    letters_in = []
    for word in word_list:
        for i in word:
            if i not in letters_in:
                letters_in.append(i)
    return letters_in

def create_spectral_bank(word_signals, f):
    spectral_bank = {}
    for word, signal in word_signals.items():
        freq, spectrum = apply_fourier_transform(signal, f)
        spectral_bank[word] = (freq, np.abs(spectrum))
    return spectral_bank

def correlate_spectra(spectrum1, spectrum2):
    min_length = min(len(spectrum1), len(spectrum2))
    spectrum1 = spectrum1[:min_length]
    spectrum2 = spectrum2[:min_length]
    correlation = np.correlate(spectrum1, spectrum2, mode='valid')
    return np.max(correlation)

def compare_spectral_properties(spectrum1, spectrum2):
    mean_diff = np.abs(np.mean(spectrum1) - np.mean(spectrum2))
    variance_diff = np.abs(np.var(spectrum1) - np.var(spectrum2))
    energy_diff = np.abs(np.sum(spectrum1 ** 2) - np.sum(spectrum2 ** 2))
    return mean_diff, variance_diff, energy_diff

def cosine_similarity(spectrum1, spectrum2):
    min_length = min(len(spectrum1), len(spectrum2))
    spectrum1 = spectrum1[:min_length]
    spectrum2 = spectrum2[:min_length]
    return np.dot(spectrum1, spectrum2) / (np.linalg.norm(spectrum1) * np.linalg.norm(spectrum2))

def visualize_spectral_comparison(freq, spectrum1, spectrum2, word_label):
    plt.figure(figsize=(12, 6))
    plt.plot(freq, spectrum1, label='Spektrum vstupního slova')
    plt.plot(freq, spectrum2, label=f'Spektrum slova "{word_label}" z banky')
    plt.legend()
    plt.show()

# Načtení signálu ze souboru
signal_path = './InputData/Signal1.txt'
signal = load_signal(signal_path)
threshold = 0.5

# Identifikace slov v signálu
words = identify_words(signal, threshold)

# Vizualizace signálu s identifikovanými slovy
visualize_signal(signal, apply_hilbert(apply_hamming(signal)), words)

# Výpis identifikovaných slov
print("Identifikovaná slova - indexy:", words, '\n', "Počet slov:", len(words))

# Extrakce hodnot jednotlivých slov ze signálu
sorted_words = sort_values(signal, words)

# Získání seznamu unikátních písmen (pro informaci)
letters = get_letters()
print("Unikátní písmena ve slovníku:", letters, "Počet písmen:", len(letters))

# Generování syntetických signálů pro každé slovo ve slovníku (pro demonstrační účely)
word_signals = {}
for i, word in enumerate(word_list):
    duration = 1.0  # Délka signálu v sekundách
    t = np.linspace(0, duration, int(f * duration), endpoint=False)
    frequency = 300 + i * 50  # Různá frekvence pro každé slovo
    word_signal = np.sin(2 * np.pi * frequency * t)
    word_signals[word] = word_signal

# Vytvoření banky frekvenčních spekter pro jednotlivá slova
spectral_bank = create_spectral_bank(word_signals, f)

# Vytvoření slovníku pro uložení frekvenčních spekter identifikovaných slov
fourier = {}

# Porovnání spekter identifikovaných slov se spektrem slov ve slovníku
recognized_words = {}
for index, values in sorted_words.items():
    # Výpočet Fourierovy transformace identifikovaného slova
    freq, spect = apply_fourier_transform(values, f)
    fourier[index] = freq, spect

    # Inicializace pro porovnání
    max_correlation = -np.inf
    best_match = None

    input_spectrum = np.abs(spect)

    for word, (bank_freq, bank_spectrum) in spectral_bank.items():
        # Interpolace spektra z banky, aby odpovídalo délce vstupního spektra
        interp_func = interp1d(bank_freq, bank_spectrum, bounds_error=False, fill_value=0)
        bank_spectrum_interp = interp_func(freq)

        # Ujistěte se, že spektra mají stejnou délku
        min_length = min(len(input_spectrum), len(bank_spectrum_interp))
        input_spectrum_trimmed = input_spectrum[:min_length]
        bank_spectrum_trimmed = bank_spectrum_interp[:min_length]

        # Výpočet korelace mezi spektry
        corr = correlate_spectra(input_spectrum_trimmed, bank_spectrum_trimmed)

        # Porovnání korelace pro nalezení nejlepší shody
        if corr > max_correlation:
            max_correlation = corr
            best_match = word

    recognized_words[index] = best_match
    print(f"Slovo {index} nejlépe odpovídá slovu '{best_match}' s korelací {max_correlation}")

    # Volitelná vizualizace spektrálního porovnání
    visualize_spectral_comparison(freq[:min_length], input_spectrum_trimmed, bank_spectrum_trimmed, best_match)

# Výpis rozpoznaných slov
print("Rozpoznaná slova:", recognized_words)
