import os
import numpy as np
import wfdb
import matplotlib.pyplot as plt
from scipy.signal import resample

# Definice funkcí


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
    output_length = len(signal) + len(kernel) - 1
    output = np.zeros(output_length)
    for i in range(output_length):
        start = max(0, i - len(kernel) + 1)
        end = min(len(signal), i + 1)
        for j in range(start, end):
            output[i] += signal[j] * kernel[i - j]
    return output


def custom_find_peaks(signal, adaptive_thresholds, min_distance):
    is_peak = (signal[1:-1] > adaptive_thresholds[1:-1]) & (signal[1:-1] > signal[:-2]) & (signal[1:-1] > signal[2:])
    peaks = np.where(is_peak)[0] + 1
    refined_peaks = [peaks[0]]
    for peak in peaks[1:]:
        if peak - refined_peaks[-1] > min_distance:
            refined_peaks.append(peak)
    return np.array(refined_peaks)


def hearth_rate(data_path):
    signals, fields = wfdb.rdsamp(data_path)
    ECG_signal = signals[:, 0]
    fs = fields['fs']
    print(f"Vzork. freq: {fs}")

    window_size_median = 3
    ECG_smoothed = median_filter(ECG_signal, window_size_median)

    window_size_adaptive = 3
    adaptive_thresholds = adaptive_threshold_median(ECG_smoothed, window_size_adaptive)

    R_peaks_indices = custom_find_peaks(ECG_smoothed, adaptive_thresholds, fs * 0.6)

    """
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
    """

    heart_rate = len(R_peaks_indices) / (len(ECG_smoothed) / fs) * 60
    print("Detekováno R-vrcholů:", len(R_peaks_indices))
    print("Tepová frekvence:", heart_rate, "bpm")

    """
    start_index = 0
    end_index = int(fs * 2)
    
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
    """

    return R_peaks_indices, ECG_smoothed, fs


def corelation_coef(signal1, signal2):
    sig1 = signal1
    sig2 = signal2

    mean_sig1 = np.mean(sig1)
    mean_sig2 = np.mean(sig2)
    numerator = np.sum((sig1 - mean_sig1) * (sig2 - mean_sig2))
    denominator = np.sqrt(np.sum((sig1 - mean_sig1) ** 2) * np.sum((sig2 - mean_sig2) ** 2))

    if denominator == 0 or np.isnan(denominator):
        correlation_coefficient = np.nan
    else:
        correlation_coefficient = numerator / denominator

    return correlation_coefficient, (sig1 - mean_sig1) * (sig2 - mean_sig2)


# Vytvoření seznamu souborů k analýze
lib_path = 'InputData'
files = os.listdir(lib_path)
drive_files = [file for file in files if file.endswith('.hea')]

# Pro každý soubor zavolejte funkci HearthRate() a uložte výsledky
results = []
for file_name in drive_files:
    file_path = os.path.join(lib_path, os.path.splitext(file_name)[0])
    r_peaks, sig_smoothed, fs = hearth_rate(file_path)
    results.append((r_peaks, sig_smoothed, fs))

# Zachovejte informace o frekvenci vzorkování a identifikovaných R-vrcholů
for r_peaks, sig_smoothed, fs in results:
    print("Frekvence vzorkování:", fs)
    print("Detekované R-vrcholy:", len(r_peaks))

# Získání dominantní vzorkovací frekvence z listu results
dominant_fs = np.argmax(np.bincount([fs for _, _, fs in results]))
print(f"float: {dominant_fs}")
dominant_fs = int(round(dominant_fs))  # Zaokrouhlení na nejbližší celé číslo
print(dominant_fs)

# Resampling signálů na dominantní vzorkovací frekvenci
resampled_signals = []
for r_peaks, sig_smoothed, fs in results:
    resampled_signal = resample(sig_smoothed, int(len(sig_smoothed) * dominant_fs / fs))
    resampled_signals.append(resampled_signal)

# Zarovnání signálů podle prvního R-peaku
aligned_signals = []
for resampled_signal, (r_peaks, _, _) in zip(resampled_signals, results):
    first_r_peak = r_peaks[0]
    aligned_signal = resampled_signal[first_r_peak:]
    aligned_signals.append(aligned_signal[:2000])

# Vykreslení signálů
plt.figure(figsize=(12, 6))
for i, aligned_signal in enumerate(aligned_signals):
    plt.plot(aligned_signal, label=f'Signál {i+1}')
plt.title('Resamplované a zarovnané signály')
plt.xlabel('Vzorky')
plt.ylabel('Amplituda')
plt.legend()
plt.grid(True)
plt.show()

corelation_coef, corelation_points = corelation_coef(aligned_signals[3], aligned_signals[4])
print("Korelační koef:", corelation_coef)
print("Korelační koef body:", corelation_points)

plt.plot(corelation_points,"r")
plt.show()