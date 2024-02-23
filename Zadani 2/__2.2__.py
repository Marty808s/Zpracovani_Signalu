import numpy as np
import matplotlib.pyplot as plt
import wfdb
import os

# Cesta k adresáři se vstupními daty
lib_path = './InputData/'

# Načtení seznamu souborů s příponou '.hea'
files = os.listdir(lib_path)
drive_files = [file for file in files if file.endswith('.hea')]


# Funkce pro převzorkování signálu na danou vzorkovací frekvenci
def resample_ecg_signal(signal, old_fs, new_fs):
    num_samples_new = int(len(signal) * (new_fs / old_fs))
    resampled_signal = signal[:num_samples_new]
    return resampled_signal


# Funkce pro nalezení pozice prvního dominantního R peaku
def find_first_r_peak(signal, threshold):
    peaks = []
    for i in range(1, len(signal) - 1):
        if signal[i] > threshold and signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
            peaks.append(i)
    if peaks:
        return peaks[0]
    else:
        return None


# Funkce pro provedení korelační analýzy mezi dvěma signály
def correlate_signals(signal1, signal2, max_lag):
    return np.correlate(signal1, signal2, mode='full')[:max_lag]


# Normalizace korelační funkce
def normalize_correlation(correlation_result):
    correlation_result -= np.min(correlation_result)
    correlation_result /= np.max(correlation_result)
    return correlation_result


# Výpočet dominantní vzorkovací frekvence
dominantni_fs = 15.5

# Načtení a převzorkování signálů, nalezení pozice prvního R peaku
resampled_signals = []
first_r_peak_positions = []

for file in drive_files:
    record_name = os.path.splitext(file)[0]
    signals, fields = wfdb.rdsamp(os.path.join(lib_path, record_name))
    ecg_signal = signals[:, 0]
    ecg_signal = ecg_signal[:5000]

    fs = fields['fs']

    # Převzorkování signálu na dominantní vzorkovací frekvenci
    resampled_signal = resample_ecg_signal(ecg_signal, fs, dominantni_fs)
    resampled_signals.append(resampled_signal)

    # Nalezení pozice prvního R peaku
    threshold = 0.6 * max(resampled_signal)
    first_r_peak_position = find_first_r_peak(resampled_signal, threshold)
    first_r_peak_positions.append(first_r_peak_position)

# Určení nejnižší pozice prvního R peaku pro posunutí signálů na stejný počátek
min_r_peak_position = min(first_r_peak_positions)

# Posunutí signálů na stejný počátek
aligned_signals = [np.roll(signal, -min_r_peak_position) for signal in resampled_signals]

# Korelační analýza mezi prvními třemi signály
correlation_results = []
max_lag = min(5000, min([len(signal) for signal in aligned_signals])) // 2
for i in range(3):
    for j in range(i + 1, 3):
        correlation_result = correlate_signals(aligned_signals[i], aligned_signals[j], max_lag)
        correlation_result = normalize_correlation(correlation_result)
        correlation_results.append(correlation_result)

# Vykreslení normalizovaných korelačních funkcí
plt.figure(figsize=(10, 6))
for i, correlation_result in enumerate(correlation_results):
    plt.plot(correlation_result, label=f'Correlation {i + 1}')

plt.xlabel('Lag')
plt.ylabel('Normalized Correlation')
plt.title('Cross-correlation between first three ECG signals')
plt.legend()
plt.grid(True)
plt.show()
