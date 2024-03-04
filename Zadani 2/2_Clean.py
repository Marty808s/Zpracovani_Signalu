import numpy as np
import matplotlib.pyplot as plt
import wfdb
import os

from scipy.signal import medfilt

lib_path = './InputData/'
files = os.listdir(lib_path)
drive_files = [file for file in files if file.endswith('.hea')]

print(drive_files) #<- mám všechny .hea soubory

#Build <- s prvním filem
one_file = drive_files[0]
record_name = os.path.splitext(one_file)[0]
signals, fields = wfdb.rdsamp(os.path.join(lib_path, record_name))
ECG_signal = signals[:, 0]
fs = fields['fs']
distance = fs * 0.6
print(f"Frekvence: {fs} | Distance: {distance}")


"""
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
"""

def find_peaks_numeric(signal, threshold, distance=distance):
    peaks = []
    last_peak = 0
    for i in range(1, len(signal) - 1):
        if signal[i] > threshold[i] and signal[i] > signal[i-1] and signal[i] > signal[i+1]:
            if i - last_peak > distance:
                peaks.append(i)
                last_peak = i
    return peaks


def adaptive_threshold_median(signal, window_size):
    thresholds = []
    for i in range(len(signal)):
        start = max(0, i - window_size)
        end = min(len(signal), i + window_size)
        threshold = np.median(signal[start:end])
        thresholds.append(threshold)
    return np.array(thresholds)


ecg_cut = ECG_signal[:3000]

# Použití mediánového filtru k vyhlazení signálu
window_size_median = 3
ECG_smoothed = medfilt(ecg_cut, kernel_size=window_size_median)

# Použití adaptivního prahování založeného na mediánu
window_size_adaptive = 3
adaptive_thresholds = adaptive_threshold_median(ECG_smoothed, window_size_adaptive)

# Detekce peaků
peaks = find_peaks_numeric(ECG_smoothed, adaptive_thresholds, distance)

heart_rate = len(peaks) / (len(ECG_smoothed) / fs) * 60
print("Detekováno R-vrcholů:", len(peaks))
print("Tepová frekvence:", heart_rate, "bpm")

plt.plot(ecg_cut, 'g')
plt.plot(ECG_smoothed, label='EKG signál {}'.format(record_name))
plt.plot(peaks, ECG_smoothed[peaks], 'r.', markersize=10, label='Detekované peaky {}'.format(record_name))
plt.show()