import numpy as np
import wfdb
from scipy.signal import find_peaks, butter, filtfilt
import matplotlib.pyplot as plt


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def detect_r_peaks(ECG_signal, fs):
    # Nastavení prahové hodnoty pro detekci R-vrcholů
    threshold = 0.6 * np.max(ECG_signal)

    # Detekce R-vrcholů
    R_peaks_indices, _ = find_peaks(ECG_signal, height=threshold, distance=fs * 0.6)

    return R_peaks_indices


def calculate_heart_rate(R_peaks_indices, fs):
    # Vypočet tepové frekvence v bpm (beats per minute)
    heart_rate = len(R_peaks_indices) / (len(ECG_signal) / fs) * 60

    return heart_rate


# Načtení dat EKG signálu
data_path = 'InputData/drive01'
signals, fields = wfdb.rdsamp(data_path)
ECG_signal = signals[:, 0]
fs = fields['fs']

# Definice dolní a horní řezové frekvence pro filtr Butterworth
lowcut = 0.5/(fs/2)  # Dolní řezová frekvence v Hz
highcut = 10.0/(fs/2) # Horní řezová frekvence v Hz

# Pročištění signálu pomocí průchodového pásmového filtru Butterworth
ECG_filtered = butter_bandpass_filter(ECG_signal, lowcut, highcut, fs)

# Detekce R-vrcholů
R_peaks_indices = detect_r_peaks(ECG_filtered, fs)

# Výpočet tepové frekvence
heart_rate = calculate_heart_rate(R_peaks_indices, fs)

# Vykreslení původního a filtrovaného signálu
plt.figure(figsize=(10, 6))
plt.plot(ECG_signal, label='Původní EKG signál')
plt.plot(ECG_filtered, label='Filtrovaný EKG signál')
plt.scatter(R_peaks_indices, ECG_filtered[R_peaks_indices], color='red', label='Detekované R-vrcholy')
plt.title('Detekce R-vrcholů a filtrování EKG signálu')
plt.xlabel('Vzorky')
plt.ylabel('Hodnota')
plt.legend()
plt.show()

print("Detekováno R-vrcholů:", len(R_peaks_indices))
print("Tepová frekvence:", heart_rate, "bpm")
