import numpy as np
import matplotlib.pyplot as plt

# Parametry signálu
amplitude = 1.0  # Amplituda
frequency = 5.0  # Frekvence (počet oscilací za sekundu)
phase = 0.0      # Fáze (posunutí signálu)

# Časové body
t = np.linspace(0, 1, 1000)  # 1000 bodů od 0 do 1 sekundy

# Generování sinusového signálu
sin_signal = amplitude * np.sin(2 * np.pi * frequency * t + phase)

# Vykreslení signálu
plt.plot(t, sin_signal)
plt.title('Sinusový signál')
plt.xlabel('Čas (s)')
plt.ylabel('Amplituda')
plt.grid(True)
plt.show()

# Parametry signálu
amplitude = 1.0  # Amplituda
frequency = 2.0  # Frekvence (počet změn polarity za sekundu)

# Generování obdélníkového signálu
square_signal = amplitude * np.sign(np.sin(2 * np.pi * frequency * t))

# Vykreslení signálu
plt.plot(t, square_signal)
plt.title('Obdélníkový signál')
plt.xlabel('Čas (s)')
plt.ylabel('Amplituda')
plt.grid(True)
plt.show()

# Parametry signálu
amplitude = 1.0  # Amplituda
frequency = 3.0  # Frekvence (počet změn polarity za sekundu)

# Generování pilového signálu
sawtooth_signal = amplitude * (2 * (t * frequency - np.floor(t * frequency + 0.5)))

# Vykreslení signálu
plt.plot(t, sawtooth_signal)
plt.title('Pilový signál')
plt.xlabel('Čas (s)')
plt.ylabel('Amplituda')
plt.grid(True)
plt.show()

