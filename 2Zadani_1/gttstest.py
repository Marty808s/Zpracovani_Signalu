import os
import gtts
import soundfile as sf
import numpy as np

word_list = ('time', 'prepare', 'solution', 'make', 'mistake', 'no', 'the', 'probable', 'long', 'lecture', 'method',
             'disaster', 'fail', 'work', 'advice', 'idea', 'succeed', 'easy', 'is', 'for', 'give')

for word in word_list:
    wav_name = f"{word}.wav"
    path = "./Samples/gtts"
    mp3_name = f"{word}.mp3"
    path_to_save_mp3 = os.path.join(path, mp3_name)
    path_to_save_wav = os.path.join(path, wav_name)

    if not os.path.exists(path):
        os.makedirs(path)

    model = gtts.gTTS(word_list[0])
    model.save(path_to_save_mp3)

    sample_rate = 22050
    duration = 2
    t = np.linspace(0., duration, int(sample_rate * duration))
    waveform = 0.5 * np.sin(2 * np.pi * 440 * t)

    # Uložit generovaný WAV soubor
    sf.write(path_to_save_wav, waveform, sample_rate)

    print(f"Soubory {mp3_name} a {wav_name} byly uloženy.")
