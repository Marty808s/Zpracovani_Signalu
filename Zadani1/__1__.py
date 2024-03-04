import os
import numpy as np
import wfdb
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, medfilt

# TODO: manual : Numpy.Median (?), scipy.findpeaks


def adaptive_threshold_median(signal, window_size):
    thresholds = []
    for i in range(len(signal)):
        start = max(0, i - window_size)
        end = min(len(signal), i + window_size)
        threshold = np.median(signal[start:end])
        thresholds.append(threshold)
    return np.array(thresholds)


def median_filter(signal, window_median):
    kernel = np.ones(window_median) / window_median
    smoothed_signal = convolution(signal, kernel)
    return smoothed_signal


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


def HearthRate(data_path):
    signals, fields = wfdb.rdsamp(data_path)
    ECG_signal = signals[:, 0]  # Získání signálu 'ECG'
    fs = fields['fs']
    print(f"Vzork. freq: {fs}")

    # Funkce pro adaptivní prahování založené na mediánu

    #  mediánového filtru k vyhlazení signálu
    window_size_median = 3
    ECG_smoothed = median_filter(ECG_signal, window_size_median)

    # Použití adaptivního prahování založeného na mediánu
    window_size_adaptive = 3
    adaptive_thresholds = adaptive_threshold_median(ECG_smoothed, window_size_adaptive)

    # Detekce R-vrcholů s použitím adaptivního prahu
    R_peaks_indices, _ = find_peaks(ECG_smoothed, height=adaptive_thresholds, distance=fs * 0.6)

    # Vykreslení signálu s detekovanými R-vrcholy a adaptivním prahem
    plt.figure(figsize=(12, 6))
    plt.plot(ECG_smoothed, label='Vyhlazený EKG signál')
    plt.plot(R_peaks_indices, ECG_smoothed[R_peaks_indices], 'ro', label='Detekované R-vrcholy')
    plt.plot(adaptive_thresholds, label='Adaptivní prah', linestyle='--')
    plt.title('Vyhlazený EKG signál s detekovanými R-vrcholy a adaptivním prahem')
    plt.xlabel('Vzorky')
    plt.ylabel('Amplituda')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Výpočet tepové frekvence
    heart_rate = len(R_peaks_indices) / (len(ECG_smoothed) / fs) * 60
    print("Detekováno R-vrcholů:", len(R_peaks_indices))
    print("Tepová frekvence:", heart_rate, "bpm")

    start_index = 0
    end_index = int(fs * 2)

    # Vykreslení 2sekundového úseku signálu s detekovanými R-vrcholy a adaptivním prahem
    plt.figure(figsize=(12, 6))
    plt.plot(ECG_smoothed[start_index:end_index], label='Vyhlazený EKG signál')
    plt.plot(R_peaks_indices[(R_peaks_indices >= start_index) & (R_peaks_indices < end_index)],
             ECG_smoothed[R_peaks_indices[(R_peaks_indices >= start_index) & (R_peaks_indices < end_index)]],
             'ro', label='Detekované R-vrcholy')
    plt.plot(range(start_index, end_index), adaptive_thresholds[start_index:end_index],
             label='Adaptivní prah', linestyle='--')
    plt.title('2sekundový úsek vyhlazeného EKG signálu s detekovanými R-vrcholy a adaptivním prahem')
    plt.xlabel('Vzorky')
    plt.ylabel('Amplituda')
    plt.legend()
    plt.grid(True)
    plt.show()


lib_path = 'InputData'
files = os.listdir(lib_path)
drive_files = [file for file in files if file.endswith('.hea')]

print(drive_files)

for item in drive_files:
    file_name = os.path.splitext(item)[0]
    print(file_name)
    HearthRate(os.path.join(lib_path, file_name))

# TODO: - předělat buidl in funkce na origo - konvoluce!
