import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

def identify_words(signal, threshold):
    words = []

    # Aplikace Hammingova okna na signál
    hamming_window = sig.get_window('hamming', len(signal))
    signal_hamming = signal * hamming_window

    # Aplikace Hilbertovy transformace na signál pro získání analytickeho signálu
    analytic_signal = sig.hilbert(signal_hamming)
    amplitude_envelope = np.abs(analytic_signal)

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

def identify_letters(signal, rules, words):
    identified_letters = []

    for start, end in words:
        # Extrahujte část signálu pro dané slovo
        word_signal = signal[start:end]

        # Inicializujte proměnnou pro identifikované písmeno ve slově
        identified_word = ""

        # Projděte pravidla pro jednotlivá písmena a najděte shodu pro každé písmeno ve slově
        for letter, rule in rules.items():
            frequency_range = rule['frequency_range']
            amplitude_threshold = rule['amplitude_threshold']

            # Vyčkejte příznaky (frekvenci a amplitudu) z části signálu pro dané slovo
            word_features = extract_features(word_signal)

            # Zkontrolujte shodu s pravidlem pro písmeno
            if check_match(word_features, frequency_range, amplitude_threshold):
                identified_word += letter

        identified_letters.append(identified_word)

    return identified_letters


def extract_features(signal_segment):
    # Vytvoření prázdného seznamu pro příznaky
    features = []

    # Pokud je signální segment prázdný, vrátíme prázdný seznam příznaků
    if len(signal_segment) == 0:
        return features

    # Extrahujeme několik jednoduchých příznaků z signálního segmentu
    # Například můžeme extrahovat průměr, maximální hodnotu a standardní odchylku
    mean = np.mean(signal_segment)
    maximum = np.max(signal_segment)
    std_dev = np.std(signal_segment)

    # Přidáme extrahované příznaky do seznamu příznaků
    features.extend([mean, maximum, std_dev])

    return features


def check_match(features, frequency_range, amplitude_threshold):
    # Rozbalení hodnot frekvenčního rozsahu
    min_frequency, max_frequency = frequency_range

    # Extrahování příznaků
    mean, maximum, std_dev = features

    # Kontrola shody mezi extrahovanými příznaky a pravidly pro dané písmeno
    # Například kontrola, zda se průměr nachází v určeném rozsahu frekvencí
    # a zda maximální hodnota překračuje stanovený prah amplitudy
    if min_frequency <= mean <= max_frequency and maximum >= amplitude_threshold:
        return True
    else:
        return False
def get_rules():
    rules = {
        'a': {'frequency_range': (100, 1000), 'amplitude_threshold': 0.7},
        'b': {'frequency_range': (200, 1500), 'amplitude_threshold': 0.8},
        'c': {'frequency_range': (300, 2000), 'amplitude_threshold': 0.9},
        'd': {'frequency_range': (400, 2500), 'amplitude_threshold': 0.7},
        'e': {'frequency_range': (500, 3000), 'amplitude_threshold': 0.8},
        'f': {'frequency_range': (600, 3500), 'amplitude_threshold': 0.9},
        'g': {'frequency_range': (700, 4000), 'amplitude_threshold': 0.7},
        'h': {'frequency_range': (800, 4500), 'amplitude_threshold': 0.8},
        'i': {'frequency_range': (900, 5000), 'amplitude_threshold': 0.9},
        'j': {'frequency_range': (1000, 5500), 'amplitude_threshold': 0.7},
        'k': {'frequency_range': (1100, 6000), 'amplitude_threshold': 0.8},
        'l': {'frequency_range': (1200, 6500), 'amplitude_threshold': 0.9},
        'm': {'frequency_range': (1300, 7000), 'amplitude_threshold': 0.7},
        'n': {'frequency_range': (1400, 7500), 'amplitude_threshold': 0.8},
        'o': {'frequency_range': (1500, 8000), 'amplitude_threshold': 0.9},
        'p': {'frequency_range': (1600, 8500), 'amplitude_threshold': 0.7},
        'q': {'frequency_range': (1700, 9000), 'amplitude_threshold': 0.8},
        'r': {'frequency_range': (1800, 9500), 'amplitude_threshold': 0.9},
        's': {'frequency_range': (1900, 10000), 'amplitude_threshold': 0.7},
        't': {'frequency_range': (2000, 10500), 'amplitude_threshold': 0.8},
        'u': {'frequency_range': (2100, 11000), 'amplitude_threshold': 0.9},
        'v': {'frequency_range': (2200, 11500), 'amplitude_threshold': 0.7},
        'w': {'frequency_range': (2300, 12000), 'amplitude_threshold': 0.8},
        'x': {'frequency_range': (2400, 12500), 'amplitude_threshold': 0.9},
        'y': {'frequency_range': (2500, 13000), 'amplitude_threshold': 0.7},
        'z': {'frequency_range': (2600, 13500), 'amplitude_threshold': 0.8}}
    return rules

# Načtení signálu ze souboru
with open('InputData/Signal1.txt', 'r') as file:
    signal_values = [float(line.strip()) for line in file.readlines()]

# Převod listu na numpy array
signal = np.array(signal_values)

# Identifikace slov v signálu s prahovou hodnotou 0.5 pomocí Hammingova okna
words = identify_words(signal, threshold=0.5)

# Načtení pravidel pro identifikaci písmen
rules = get_rules()

# Identifikace písmen ve slovech
identified_letters = identify_letters(signal, rules, words)

print("Identifikovaná písmena ve slovech:", identified_letters)

plt.plot(signal_values)
plt.show()
