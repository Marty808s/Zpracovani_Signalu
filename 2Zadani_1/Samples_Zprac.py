import numpy as np
import matplotlib.pyplot as plt
import sympy as smp
import wave, struct

t, f = smp.symbols('t, f', real=True)
k = smp.symbols('k', real=True, positive=True)
x = smp.exp(-k * t ** 2) * k * t

freq = 22050

def convolution(signal, kernel):
    # Výstupní signál bude mít délku len(signal) + len(kernel) - 1
    output_length = len(signal) + len(kernel) - 1
    output = np.zeros(output_length)
    return output

def apply_fourier_transform(signal, f=f):
    spectrum = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(signal), d=1/f)
    return frequencies, spectrum


def visualize_signal(signal, amplitude_envelope, words):
    plt.plot(signal, label='Signál')
    plt.plot(amplitude_envelope, label='Amplitudový obal')
    plt.axhline(y=0.5, color='r', linestyle='-')
    for start, end in words:
        plt.axvspan(start, end, color='gray', alpha=0.5)
    plt.legend()
    plt.show()


def cut_signal(signal,indexes):
    A = []
    for i in indexes:
        y = signal[i[0]:i[1]]
        A.append(y)
    return A

indexes = [
    (221000, 260000),
    (380000, 430000),
    (545000, 600000),
    (730000, 770000),
    (910000, 960000),
    (1080000, 1130000),
    (1255000, 1310000),
    (1430000, 1490000),
    (1615000, 1670000),
    (1790000, 1840000),
    (1970000, 2020000),
    (2180000, 2220000),
    (2350000, 2400000),
    (2530000, 2580000),
    (2710000, 2750000),
    (2880000, 2940000),
    (3070000, 3130000),
    (3250000, 3310000),
    (3440000, 3500000),
    (3610000, 3670000),
    (3780000, 3850000),
    (3960000, 4030000),
    (4150000, 4210000),
    (4340000, 4400000),
    (4500000, 4570000),
    (4670000, 4730000)]

wav_file = 'Samples/best_zaznam.wav'

threshold = 100
win_size = 20000
wavefile = wave.open(wav_file, 'r')

length = wavefile.getnframes()

data = []
num_channels = wavefile.getnchannels()
sample_width = wavefile.getsampwidth()

for i in range(0, length):
    wavedata = wavefile.readframes(1)
    if wavedata:
        # Načtení hodnot pro každý kanál
        if sample_width == 1:  # Pokud je šířka vzorku 1 byte
            values = struct.unpack(f"{num_channels}B", wavedata)
        elif sample_width == 2:  # Pokud je šířka vzorku 2 byty
            values = struct.unpack(f"{num_channels}h", wavedata)
        else:
            raise ValueError("Unsupported sample width")

        # Přidání hodnot každého kanálu do seznamu
        for j in range(num_channels):
            data.append(values[j])
    else:
        break


plt.plot(data)
plt.show()

print(len(data))
signal = data

alphabet = cut_signal(signal,indexes)
print(len(alphabet))
print(alphabet)

for i, s in enumerate(alphabet):
    plt.plot(s, label=f'Podsignál {i+1}')
plt.legend()
plt.show()

fourier = {}

for index, values in enumerate(alphabet):
    freq, spect = apply_fourier_transform(values, freq)
    fourier[index] = freq, spect
    plt.plot(freq, np.abs(spect))
    plt.title(f'Spektrum podsignálu {index+1}')
    plt.xlabel('Frekvence')
    plt.ylabel('Amplituda')
    plt.show()

print(fourier)