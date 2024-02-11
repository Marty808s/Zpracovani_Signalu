import numpy as np
import wfdb
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, medfilt

# Načtení dat EKG signálu
data_path = 'InputData/drive07'
signals, fields = wfdb.rdsamp(data_path)
ECG_signal = signals[:, 0]  # Získání signálu 'ECG'
fs = fields['fs']
print(f"Vzork. freq: {fs}")
