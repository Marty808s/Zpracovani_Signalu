import numpy as np
import matplotlib.pyplot as plt
import wfdb

# Načtení dat EMG signálu
data_path = 'InputData/drive07'
signals, fields = wfdb.rdsamp(data_path)
EMG_signal = signals[:, 1]  # Získání signálu 'EMG'
fs = fields['fs']
print(fields)
print(f"Vzork. freq: {fs}")

print(len(EMG_signal))
plt.plot(EMG_signal[:5000])
plt.axhline(0, color="r")
plt.show()

time = round(len(EMG_signal)/fs,3)
print(time)


def integration(signal):
    total_integral = []
    for i in range(len(signal) - 1):
        res = abs(((signal[i] + signal[i + 1]) * 1/fs) / 2)
        total_integral.append(res)
    return total_integral



result = integration(EMG_signal)
#print("Integration result:", result)
plt.plot(result)
plt.show()


def window_detection(signal, width, threshold):
    windows = []
    for i in range(len(signal) - width):
        window = signal[i:i + width]
        window_range = np.max(window) - np.min(window)
        if window_range <= threshold:
            marker = np.diff(window)
            sign = np.sign(np.mean(marker))
            if sign == 1:
                print("Rostem")
            elif sign == -1:
                print("Klesáme")
            part_integ = integration(window)
            windows.append((window, part_integ))

    return windows


window_width = 100
threshold = 0.08

detected_windows = window_detection(EMG_signal[:1000], window_width, threshold)
print(len(detected_windows))


# Plot the integral within all detected windows
for window, integral_values in detected_windows:
    plt.plot(integral_values)

plt.xlabel('Sample Index')
plt.ylabel('Integral')
plt.title('Integral within detected windows')
plt.show()