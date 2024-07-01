import os
import numpy as np
import matplotlib.pyplot as plt
import wfdb

# Funkce pro načtení signálu
def load_emg_signal(file_path):
    signals, fields = wfdb.rdsamp(file_path)
    EMG_signal = signals[:, 1]  # Získání signálu 'EMG'
    fs = fields['fs']
    return EMG_signal, fs

# Funkce pro výpočet integrovaného EMG (iEMG)
def integration(signal, fs):
    total_integral = []
    for i in range(len(signal) - 1):
        res = abs(((signal[i] + signal[i + 1]) * 1/fs) / 2)
        total_integral.append(res)
    return total_integral

# Funkce pro výpočet derivace a jejího okénkování
def compute_derivative(signal, fs, width):
    derivative = np.diff(signal) * fs
    length = len(derivative)
    sublist = np.array_split(derivative, int(length/width))
    points = []
    counter = 0
    for i in sublist:
        counter += 1
        point = np.mean(i)
        points.append((point, ((counter * width) - width/2)))
    return points

# Funkce pro detekci oblastí nárůstu a poklesu aktivity
def window_detection(derivative, threshold):
    increasing_regions = [(val, idx) for val, idx in derivative if val > threshold]
    decreasing_regions = [(val, idx) for val, idx in derivative if val < -threshold]
    return increasing_regions, decreasing_regions

# Vytvoření seznamu souborů k analýze
lib_path = 'InputData'
files = os.listdir(lib_path)
drive_files = [file.replace('.hea', '') for file in files if file.endswith('.hea')]

signals_list = []
fs_list = []

for file_name in drive_files:
    file_path = os.path.join(lib_path, file_name)
    signal, fs = load_emg_signal(file_path)
    signals_list.append(signal)
    fs_list.append(fs)

# Zpracování každého signálu
threshold = 0.0001
window_width = 10

for idx, EMG_signal in enumerate(signals_list):
    fs = fs_list[idx]
    print(f"Processing signal {idx + 1}/{len(signals_list)} with sampling frequency {fs} Hz")

    # Výpočet iEMG
    sig = integration(EMG_signal, fs)

    # Výpočet derivace a detekce oblastí
    sig_der = compute_derivative(sig, fs, window_width)
    inc_indices, dec_indices = window_detection(sig_der, threshold)

    # Vizualizace
    if inc_indices:
        inc_values, inc_indices = zip(*inc_indices)
    else:
        inc_values, inc_indices = [], []

    if dec_indices:
        dec_values, dec_indices = zip(*dec_indices)
    else:
        dec_values, dec_indices = [], []

    plt.figure(figsize=(12, 6))
    plt.plot(EMG_signal, label='Raw signál')
    plt.plot(sig, '--', label='Integrovaný signál')
    if inc_indices:
        plt.plot(inc_indices, inc_values, 'ro', label='Nárůst aktivity')
    if dec_indices:
        plt.plot(dec_indices, dec_values, 'bo', label='Pokles aktivity')
    plt.title(f'EMG Signal {idx + 1}')
    plt.legend()
    plt.show()

for s in signals_list:
    plt.plot(s, label='Raw EMG signal')
    plt.show()