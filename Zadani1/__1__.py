import os
import pandas as pd
import numpy as np
import wfdb
import matplotlib.pyplot as plt
from Atr_Conv import run_batch

get_batch = run_batch()

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

    # Pro každý výstupní bod provedeme sum; u součinů odpovídajících vstupních bodů signálu a jádra
    for i in range(output_length):
        start = max(0, i - len(kernel) + 1)
        end = min(len(signal), i + 1)
        for j in range(start, end):
            output[i] += signal[j] * kernel[i - j]

    return output


def custom_find_peaks(signal, adaptive_thresholds, min_distance):
    is_peak = (signal[1:-1] > adaptive_thresholds[1:-1]) & (signal[1:-1] > signal[:-2]) & (signal[1:-1] > signal[2:])
    peaks = np.where(is_peak)[0]  # Add 1 to adjust for the shifted index

    # Refine peaks based on the minimum distance
    refined_peaks = [peaks[0]]
    for peak in peaks[1:]:
        if peak - refined_peaks[-1] > min_distance:
            refined_peaks.append(peak)

    return np.array(refined_peaks)


def HearthRate(data_path):
    signals, fields = wfdb.rdsamp(data_path)
    ECG_signal = signals[:, 0]  # Získání signálu 'ECG'
    fs = fields['fs']
    print(f"Vzork. freq: {fs}")

    #ECG_signal = ECG_signal[:100000]# omezení pro testování

    # Funkce pro adaptivní prahování založené na mediánu

    #  mediánového filtru k vyhlazení signálu
    window_size_median = 3
    ECG_smoothed = median_filter(ECG_signal, window_size_median)

    # Použití adaptivního prahování založeného na mediánu
    window_size_adaptive = 3
    adaptive_thresholds = adaptive_threshold_median(ECG_smoothed, window_size_adaptive)

    # Detekce R-vrcholů s použitím adaptivního prahu
    R_peaks_indices = custom_find_peaks(ECG_smoothed, adaptive_thresholds, fs * 0.6)

    # Vykreslení signálu s detekovanými R-vrcholy a adaptivním prahem
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
    #plt.show()
    """
    # Výpočet tepové frekvence
    heart_rate = round(len(R_peaks_indices) / (len(ECG_smoothed) / fs) * 60, 0) #zaokrouhlení na celé BPM
    print("Detekováno R-vrcholů:", len(R_peaks_indices))
    print("Tepová frekvence:", heart_rate, "bpm")

    start_index = 0
    end_index = int(fs * 2)

    # Vykreslení 2 sekundového úseku signálu s detekovanými R-vrcholy a adaptivním prahem
    """
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
    """
    return heart_rate

# Funkce pro porovnání výsledné tepové frekvence s referenční hodnotou - rozdíl v bpm
def compare(bpm1, bpm2):
    dif_dict = {}
    for id in bpm1.keys():
        if id in bpm2:
            diff = bpm1[id] - bpm2[id]
            print(f"Rozdíl mezi vypočítanou a referenční hodnotou {id}: {diff} bpm")
            dif_dict[id] = diff
    return dif_dict


lib_path = 'Zadani1/TestData/test_data'
csv_path = 'Zadani1'

files = os.listdir(lib_path)
drive_files = [file for file in files if file.endswith('.hea')]

result_dict = {}

for item in drive_files:
    file_name = os.path.splitext(item)[0]
    print(file_name)
    res = HearthRate(os.path.join(lib_path, file_name))
    result_dict[file_name] = res


df_output = pd.DataFrame.from_dict(result_dict, orient='index', columns=['bpm'])
df_output.reset_index(inplace=True)
df_output.columns = ['id', 'bpm']
df_output.to_csv(os.path.join(csv_path, 'out.csv'), index=False)
print(df_output)


comparsion = compare(result_dict, get_batch)
print(comparsion)

plt.figure(figsize=(12, 6))
plt.plot(comparsion.keys(), comparsion.values(), marker='o', label='Diference mezi vypočítanou a referenční hodnotou BPM')
plt.axhline(y=0, color='r', linestyle='--', label='y = 0')
plt.grid(True)
plt.legend()
plt.title('Porovnání vypočítané a referenční tepové frekvence')
plt.xlabel('ID záznamu')
plt.ylabel('Rozdíl BPM')
plt.show()

