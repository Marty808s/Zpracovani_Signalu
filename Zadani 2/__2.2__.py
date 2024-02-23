import numpy as np
import matplotlib.pyplot as plt
import wfdb
import os

lib_path = './InputData/'
files = os.listdir(lib_path)
drive_files = [file for file in files if file.endswith('.hea')]

def resample_ecg_signal(signal, old_fs, new_fs):
    num_samples_new = int(len(signal) * (new_fs / old_fs))
    resampled_signal = signal[:num_samples_new]
    return resampled_signal

def find_peaks(signal, threshold):
    peaks = []
    for i in range(1, len(signal) - 1):
        if signal[i] > threshold and signal[i] > signal[i-1] and signal[i] > signal[i+1]:
            peaks.append(i)
    return peaks

# Inicializace figure
plt.figure(figsize=(10, 6))

# Volba dominantní vzorkovací frekvence
dominantni_fs = 15.5

# Náhradní barvy pro zobrazení různých signálů
colors = ['b', 'g', 'r', "y"]

# Seznam pro uložení posunutých signálů
shifted_signals = []

# Načtení, převzetí na dominantní frekvenci, omezení délky signálů a posunutí podle prvního R-peaku
for i, file in enumerate(drive_files[:4]):
    record_name = os.path.splitext(file)[0]
    signals, fields = wfdb.rdsamp(os.path.join(lib_path, record_name))
    ecg_signal = signals[:, 0]
    fs = fields['fs']
    resampled_signal = resample_ecg_signal(ecg_signal, fs, dominantni_fs)
    peaks = find_peaks(resampled_signal, threshold=0.6 * max(resampled_signal))
    first_peak_index = peaks[0]
    shifted_signal = np.roll(resampled_signal, -first_peak_index)
    shifted_signals.append(shifted_signal[:500])

# Zobrazení posunutých signálů na jednom grafu
for i, shifted_signal in enumerate(shifted_signals):
    plt.plot(shifted_signal, label='EKG signál {}'.format(i+1), color=colors[i])

# Nastavení popisků a titulu
plt.xlabel('Čas (s)')
plt.ylabel('Amplituda')
plt.title('Porovnání posunutých EKG signálů')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
