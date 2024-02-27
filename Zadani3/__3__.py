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


def compute_derivative(signal, fs): #derivace pole, pak podle width udělám list -> mean -> přepíšeme list do 1 hodnoty - vracíme jako hodnotu pro okno
    derivative = np.diff(signal) * fs
    return derivative


def window_detection(derivative, threshold):
    increasing_regions = np.where(derivative > threshold)[0]
    decreasing_regions = np.where(derivative < -threshold)[0]
    return increasing_regions, decreasing_regions

threshold = 0.08

sig = integration(EMG_signal)
plt.plot(sig)
plt.show()

sig_der = compute_derivative(sig,fs)
print(f"Sig_der: {sig_der}")
detected_win = window_detection(sig_der,threshold)


print(len(detected_win[0]))
print(len(detected_win[1]))

inc_indices = detected_win[0]
dec_indices = detected_win[1]

print("inc_indices:", inc_indices)
print("dec_indices:", dec_indices)

def sort_values(sig_input, inc_indices, dec_indices):
    inc = np.take(sig_input, inc_indices)
    dec = np.take(sig_input, dec_indices)
    return inc, dec

sorted_val = sort_values(sig, inc_indices, dec_indices)

plt.plot(EMG_signal)  # Vykreslit vstupní signál EMG

# Vykreslit body pro detekci nárůstu aktivity
plt.plot(inc_indices, EMG_signal[inc_indices], 'ro', label='Nárůst aktivity')

# Vykreslit body pro detekci poklesu aktivity
plt.plot(dec_indices, EMG_signal[dec_indices], 'bo', label='Pokles aktivity')

plt.legend()  # Přidat legendu
plt.show()