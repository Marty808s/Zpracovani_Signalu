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


def compute_derivative(signal, fs, width): #derivace pole, pak podle width udělám list -> mean -> přepíšeme list do 1 hodnoty - vracíme jako hodnotu pro okno
    width = width
    derivative = np.diff(signal) * fs
    print(f"Length derivation {len(derivative)}")
    sublist = np.array_split(derivative,width)
    print(f"Length sublist {len(sublist)}")
    points = []

    counter = 0
    for i in sublist:
        counter +=1
        point = np.sum(i)
        points.append((point,((counter*width)-width/2)))

    return points


def window_detection(derivative, threshold):
    increasing_regions = [(val, idx) for val, idx in derivative if val > threshold]
    decreasing_regions = [(val, idx) for val, idx in derivative if val < -threshold]
    return increasing_regions, decreasing_regions

threshold = 0.008

sig = integration(EMG_signal[:10000])
plt.plot(sig)
plt.show()

sig_der = compute_derivative(sig,fs,100)
inc_indices, dec_indices = window_detection(sig_der,threshold)


print(f"Sig_der: {sig_der}")
print(f"Length Sig_der: {len(sig_der)}")
print("inc_indices:", inc_indices)
print("length inc_indices:", len(inc_indices))
print("dec_indices:", len(inc_indices))
print("length dec_indices:", len(dec_indices))


def sort_values(sig_input, inc_indices, dec_indices):
    inc = np.take(sig_input, inc_indices)
    dec = np.take(sig_input, dec_indices)
    return inc, dec
#sorted_val = sort_values(sig, inc_indices, dec_indices)


inc_values, inc_indices = zip(*inc_indices)

print(inc_values)
print(inc_indices)

dec_values, dec_indices = zip(*dec_indices)

plt.plot(EMG_signal[:10000], label='Raw signál')
plt.plot(sig,'--', label='Integrovaý signál')  # Vykreslit vstupní signál EMG
plt.plot(inc_indices, inc_values, 'ro', label='Nárůst aktivity')
plt.plot(dec_indices, dec_values, 'bo', label='Pokles aktivity')

plt.legend()  # Přidat legendu
plt.show()