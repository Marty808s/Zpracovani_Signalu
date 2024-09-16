import os
import wfdb
import numpy as np

def calc_bpm(atr_data):
    sample_rate = 100000
    r_times = atr_data.sample[:sample_rate] / atr_data.fs
    rr_distance = np.diff(r_times)
    mean_rr = np.mean(rr_distance)
    bpm = round(60 / mean_rr, 0)
    return bpm

def read_atr(data_path):
    an = wfdb.rdann(data_path, 'atr')
    if an:
        return an
    else:
        raise ValueError("Nemám .atr data")

    
def run_batch():
    path = 'Zadani1/TestData/test_data'

    files = os.listdir(path)

    files_to_read = [file for file in files if file.endswith('.atr')]
    print(files_to_read)

    result_dict = {}

    if files_to_read:
        for file in files_to_read:
            file_name = file.split('.')[0]
            print(f"file name: {file_name}")
            path_to_file = os.path.join(path, file_name)
            print(f"path: {path_to_file}")
            an = read_atr(path_to_file)
            bpm = calc_bpm(an)
            print(bpm)
            result_dict[file_name] = bpm
    else:
        print("Nemám .atr data")

    print(result_dict)
    return result_dict

#res = run_batch()
#print(res)
