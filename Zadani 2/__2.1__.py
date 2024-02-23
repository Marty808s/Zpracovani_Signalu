import numpy as np
import matplotlib.pyplot as plt
import wfdb
import os

lib_path = './InputData/'
files = os.listdir(lib_path)
drive_files = [file for file in files if file.endswith('.hea')]

#print(drive_files)

def get_fs_values():
    fs_values = {}
    for file_name in drive_files:
        record_name = os.path.splitext(file_name)[0]
        signals, fields = wfdb.rdsamp(os.path.join(lib_path, record_name))
        ECG_signal = signals[:, 0]
        fs = fields['fs']

        fs_values[record_name] = {'fs': fs, 'duration': len(ECG_signal)}

    return fs_values

fs_values = get_fs_values()
print(fs_values)

#Volba dominantní vzork. freq
dom_fs = 15.5
length = 50000 #cut na délku array signálu ECG - pro korelační funkci

def convolution(signal, kernel):
    # Výstupní signál bude mít délku len(signal) + len(kernel) - 1
    output_length = len(signal) + len(kernel) - 1
    output = np.zeros(output_length)

    # Pro každý výstupní bod provedeme sumu součinů odpovídajících vstupních bodů signálu a jádra
    for i in range(output_length):
        start = max(0, i - len(kernel) + 1)
        end = min(len(signal), i + 1)
        for j in range(start, end):
            output[i] += signal[j] * kernel[i - j]

    return output

def find_peaks_numeric(signal, threshold):
    peaks = []
    for i in range(1, len(signal) - 1):
        if signal[i] > threshold and signal[i] > signal[i-1] and signal[i] > signal[i+1]:
            peaks.append(i)
    return peaks


# Počet grafů na jedné stránce
grafy_na_strance = 3

# Inicializace figure
plt.figure(figsize=(15, 10))

# Vypočet počtu řádků (zaokrouhlení nahoru)
pocet_radku = int(np.ceil(len(drive_files) / grafy_na_strance))

for i, file in enumerate(drive_files):
    record_name = os.path.splitext(file)[0]
    signals, fields = wfdb.rdsamp(os.path.join(lib_path, record_name))
    ecg_signal = signals[:, 0]
    fs = fields['fs']
    ecg_cut = ecg_signal[:500]
    threshold = 0.6 * max(ecg_cut)

    # Detekce peaků
    peaks = find_peaks_numeric(ecg_cut, threshold)
    print(len(peaks))

    # Vykreslení EKG signálu a detekovaných peaků
    plt.subplot(pocet_radku, grafy_na_strance, i+1)
    plt.plot(ecg_cut, label='EKG signál {}'.format(record_name))
    plt.plot(peaks, ecg_signal[peaks], 'r.', markersize=10, label='Detekované peaky {}'.format(record_name))

    # Nastavení popisků a titulu
    plt.xlabel('Čas (s)')
    plt.ylabel('Amplituda')
    plt.title('Detekce peaků pomocí numerické metody s thresholdem {}'.format(threshold))
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()



