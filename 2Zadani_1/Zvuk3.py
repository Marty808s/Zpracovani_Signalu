import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.signal as sig
from decimal import Decimal
import cmath
import pandas as pd

# Vzorkovací frekvence
f = 22050

# Slovník slov
word_list = ('time', 'prepare', 'solution', 'make', 'mistake', 'no', 'the', 'probable', 'long', 'lecture', 'method',
             'disaster', 'fail', 'work', 'advice', 'idea', 'succeed', 'easy', 'is', 'for', 'give')


def load_signal(file_path):
    """
    Funkce pro načtení signálu
    :param file_path: cesta k souboru se signálem
    :return: pole, kde je po řádcích načtený signál
    """

    with open(file_path, 'r') as file:
        signal_values = [float(line.strip()) for line in file.readlines()]
    return np.array(signal_values)


def apply_hamming(signal, words):
    """
    Aplikace Hammingova okna na jednotlivá rozpoznaná slova rozpoznaná Hilbertovou tranformací. Vyhlazuje konce slov pro
    lepší analýzu frekvenčního spektra.
    :param signal: Signál v Hilbertově obálce
    :param words: Indexy trvání jednotlivých identifikovaných slov
    :return: Vyhlazený signál připravený na Fourierovu transformaci
    """
    signal_with_hamming = np.zeros_like(signal)
    for start, end in words:
        # Aplikujeme Hammingovo okno na každé detekované slovo
        word_segment = signal[start:end]
        hamming_window = sig.get_window('hamming', len(word_segment))
        word_segment_hamming = word_segment * hamming_window
        # Nahrazení původní části signálu oknem
        signal_with_hamming[start:end] = word_segment_hamming
    return signal_with_hamming


def apply_hilbert(signal):
    """
    Vypočítá obálku amplitudy signálu pomocí Hilbertovy transformace.
    :param signal: Vstupní signál
    :return: Obálka amplitudy signálu
    """
    analytic_signal = sig.hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)
    return amplitude_envelope


def apply_fourier_transform(signal, f=f):
    """
    :param signal: Vyhlazený signál pomocí Hammingova okna
    :param f: Vzorkovací frekvence signálu
    :return: Frekvenční spektrum s jednotlivými obsaženými frekvencemi v signálu
    """
    spectrum = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(signal), d=1/f)
    return frequencies, spectrum


def visualize_signal(signal, amplitude_envelope, words):
    """
    Vizualizuje vstupní signál spolu s vyhlazeným amplitudovým obalem
    :param signal: Vstupní signál
    :param amplitude_envelope: Vyhlazené (Hamming) amplitudové obaly (Hilbert)
    :param words: Jednotlivá identifikovaná slova
    """
    plt.figure(figsize=(12, 6))
    plt.plot(signal, label='Signál')
    plt.plot(amplitude_envelope, label='Amplitudový obal')
    plt.axhline(y=0.5, color='r', linestyle='-')
    plt.axhline(y=threshold, color='k', linestyle='-', label='Treshold')
    for start, end in words:
        plt.axvspan(start, end, color='gray', alpha=0.5)
    plt.legend()
    plt.show()


def identify_words(signal, threshold):
    """
    Identifikace jednotlivých slov pomocí přechodu přes definovaný treshold
    :param signal: Vstupní signál
    :param threshold: Námi zvolený treshold
    :return: Indexy jednotlivých slov - začátek, konec
    """
    words = []
    # Aplikace Hilbertovy transformace na signál pro získání analytického signálu
    amplitude_envelope = apply_hilbert(signal)
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
    """
    Extrakce jednotlivých slov
    :param signal: Vstupní signál
    :param indexes: Identifikovaná slova
    :return: Hodnoty signálu z indexů rozpoznaných slov
    """
    vals = {}
    for num, i in enumerate(indexes):
        start_index, end_index = i
        sorted_val = signal[start_index:end_index]
        vals[num] = sorted_val
    return vals


def correlate_spectra(spectrum1, spectrum2):
    """
    Korelace spekter signálů
    :param spectrum1: Vstupní signál ze zadání
    :param spectrum2: Signál z nagenerované banky
    :return: Maximální korelační koeficient
    """
    spectrum1 = spectrum1
    spectrum2 = spectrum2
    correlation = np.correlate(spectrum1, spectrum2, mode='valid')
    return np.max(correlation)


def load_bank():
    lib_path = './Samples/gtts'
    drive_files = [file for file in os.listdir(lib_path) if file.endswith('.txt')]

    fourier = {}
    for item in drive_files:
        file_name = os.path.splitext(item)[0]
        txt_file = f"{file_name}.txt"
        path_to_file = os.path.join(lib_path, txt_file)

        sig = load_signal(path_to_file)
        freq, spec = apply_fourier_transform(sig)

        fourier[item] = freq, spec
    #   plt.plot(freq, np.abs(spec))
    #   plt.show()
    return fourier


def visualize_spectral_comparison(freq, freq2, spectrum1, spectrum2, word_label):
    plt.figure(figsize=(12, 6))
    plt.plot(freq, np.abs(spectrum1), label='Spektrum vstupního slova')
    plt.plot(freq2, np.abs(spectrum2), label=f'Spektrum slova "{word_label}" z banky')
    plt.legend()
    plt.show()


# Načtení signálu ze souboru
signal_path = './InputData/Signal3.txt'
signal = load_signal(signal_path)
spectra_bank = load_bank()
threshold = 25

# Identifikace slov v signálu
words = identify_words(signal, threshold)
smoothed = apply_hamming(apply_hilbert(signal), words)
# Vizualizace signálu s identifikovanými slovy
visualize_signal(signal, smoothed, words)

# Výpis identifikovaných slov
print("Identifikovaná slova - indexy:", words, '\n', "Počet slov:", len(words))

# Extrakce hodnot jednotlivých slov ze signálu
sorted_words = sort_values(signal, words)
# Extrakce hodnot frekvenčního spektra a jednotlivých frekvencí obsažených v signálu
frequence, fourier = apply_fourier_transform(signal, f=f)

# Provedení korelace mezi signály ze vstupu a bankou
cor = {}
for i in range(len(words)):
    freq, spect = apply_fourier_transform(sorted_words[i])
    for j in spectra_bank.keys():
        specter = spectra_bank[j]
        spect2 = specter[1]
        freq2 = specter[0]
    #    print(spect)
        cor_coef = correlate_spectra(spect, spect2)
        cor[f"{i} | {j}"] = cor_coef
    #    visualize_spectral_comparison(freq, freq2,  spect, spect2, j)
# print(cor)

# Převod komplexího čísla na decimal..
for key in cor:
    cor[key] = Decimal(cmath.polar(cor[key])[0])

data = []
for key, value in cor.items():
    sample_i, sample_j = key.split(' | ')
    data.append([sample_i, sample_j, value])

# Dataframe pro tabulku a vizualizace
df = pd.DataFrame(data, columns=['Sample_i', 'Sample_j', 'Correlation'])
csv_path = "./Samples"
df.to_csv(os.path.join(csv_path, 'cor_out.csv'), index=False)

max_cor_df = df.loc[df.groupby('Sample_i')['Correlation'].idxmax()]
final_cor = max_cor_df.sort_values("Sample_i", ascending=False)

final_cor.to_csv(os.path.join(csv_path, 'max_cor_out.csv'), index=False)

unique_samples = df['Sample_j'].unique()
colors = plt.cm.get_cmap('tab20', len(unique_samples))

color_mapping = {sample: colors(i) for i, sample in enumerate(unique_samples)}
df['color'] = df['Sample_j'].map(color_mapping)

plt.figure(figsize=(10, 6))
plt.bar(df['Sample_i'], df['Correlation'], color=df['color'])
plt.yscale('log') # Log. transformace
plt.xlabel('Sample_i')
plt.ylabel('Corr (log scale)')
plt.title('Correlation - sample_i X sample_j')
plt.xticks(df['Sample_i'])

legend_labels = [plt.Line2D([0], [0], color=color_mapping[sample], lw=4) for sample in unique_samples]
plt.legend(legend_labels, unique_samples, title='Sample_j', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()