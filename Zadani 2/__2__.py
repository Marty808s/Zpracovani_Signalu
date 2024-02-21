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

def convolution(signal, treshold, length=length ,dom_fs=dom_fs):






