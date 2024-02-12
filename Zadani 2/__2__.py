import numpy as np
import matplotlib.pyplot as plt
import wfdb
from scipy.signal import resample, correlate


def find_first_r_peak(signal):
    # Tady by měla být implementace detekce prvního R peaku, která vrátí jeho pozici
    # Pro jednoduchost předpokládejme, že index prvního R peaku je 1000
    return 1000


def compute_autocorr(signal):
    # Výpočet autokorelační funkce signálu
    autocorr = np.correlate(signal, signal, mode='full')
    return autocorr


# Načtení dat EKG signálů z vaší databáze
data_paths = ['../Zadani1/InputData/drive01', '../Zadani1/InputData/drive02',
              '../Zadani1/InputData/drive03','../Zadani1/InputData/drive07', '../Zadani1/InputData/drive08']  # Zde uveďte cesty k jednotlivým souborům
autocorrs = []

# Určení dominantní vzorkovací frekvence
dominant_fs = 0
for data_path in data_paths:
    signals, fields = wfdb.rdsamp(data_path)
    fs = fields['fs']
    if fs > dominant_fs:
        dominant_fs = fs

# Srovnání signálů na stejný počátek podle pozice prvního R peaku
for data_path in data_paths:
    signals, fields = wfdb.rdsamp(data_path)
    ECG_signal = signals[:, 0]  # Získání signálu 'ECG'
    fs = fields['fs']

    # Nalezení pozice prvního R peaku
    r_peak_pos = find_first_r_peak(ECG_signal)

    # Posunutí signálu tak, aby první R peak byl na začátku
    ECG_signal = np.roll(ECG_signal, -r_peak_pos)

    # Převzorkování signálu na dominantní vzorkovací frekvenci
    resampled_signal = resample(ECG_signal, int(len(ECG_signal) * (dominant_fs / fs)))

    # Výpočet autokorelační funkce signálu
    autocorr = compute_autocorr(resampled_signal)
    autocorrs.append(autocorr)

# Vizualizace autokorelačních funkcí
plt.figure(figsize=(10, 6))
for i, autocorr in enumerate(autocorrs):
    plt.plot(autocorr, label=f'Signál {i + 1}')
plt.xlabel('Posun')
plt.ylabel('Korelační hodnota')
plt.legend()
plt.grid(True)
plt.title('Autokorelační funkce různých EKG signálů převzorkovaných na stejnou dominantní vzorkovací frekvenci')
plt.show()
