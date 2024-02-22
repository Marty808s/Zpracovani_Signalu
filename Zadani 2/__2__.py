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

# Pan-Tompkins algoritmus pro detekci QRS komplexů
def pan_tompkins(ecg_signal, fs):
    # Nízkofrekvenční filtrace (0.5-40 Hz)
    # Implementace filtru zde
    filtered_ecg = ecg_signal  # Pro demonstrační účely necháme signál nezměněný

    # Derivace
    # Implementace derivace zde
    derivative_ecg = np.diff(filtered_ecg)

    # Squaring
    squared_ecg = derivative_ecg ** 2

    # Moving Window Integration
    # Implementace okénkové integrace zde
    window_width = int(0.150 * fs)  # 150 ms
    mwi_ecg = convolution(squared_ecg, np.ones(window_width) / window_width)

    # Adaptive Thresholding
    threshold = 0.6 * np.max(mwi_ecg)

    # Detekce vrcholů QRS
    qrs_peaks_indices = np.where(mwi_ecg > threshold)[0]

    return qrs_peaks_indices


for file in drive_files:
    record_name = os.path.splitext(file)[0]
    signals, fields = wfdb.rdsamp(os.path.join(lib_path, record_name))
    ecg_signal = signals[:, 0]
    fs = fields['fs']
    # Detekce QRS komplexů pomocí Pan-Tompkins algoritmu
    qrs_peaks_indices = pan_tompkins(ecg_signal, fs)

    # Vykreslení EKG signálu a detekovaných QRS komplexů
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(ecg_signal)) / fs, ecg_signal, label='EKG signál')
    plt.scatter(np.array(qrs_peaks_indices) / fs, ecg_signal[qrs_peaks_indices], color='red',
                label='Detekované QRS komplexy')
    plt.xlabel('Čas (s)')
    plt.ylabel('Amplituda')
    plt.title('Detekce QRS komplexů pomocí Pan-Tompkins algoritmu (s numerickou konvolucí)')
    plt.legend()
    plt.grid(True)
    plt.show()






