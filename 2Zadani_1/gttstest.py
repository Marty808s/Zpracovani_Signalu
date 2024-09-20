import os
import gtts
import numpy as np
import soundfile as sf
from Zvuk3 import load_signal, apply_fourier_transform
import matplotlib.pyplot as plt
from moviepy.editor import AudioFileClip

word_list = ('time', 'prepare', 'solution', 'make', 'mistake', 'no', 'the', 'probable', 'long', 'lecture', 'method',
             'disaster', 'fail', 'work', 'advice', 'idea', 'succeed', 'easy', 'is', 'for', 'give')
path = "./Samples/gtts"


for word in word_list:
    mp3_name = f"{word}.mp3"
    wav_name = f"{word}.wav"
    path_to_save_mp3 = os.path.join(path, mp3_name)
    path_to_save_wav = os.path.join(path, wav_name)

    model = gtts.gTTS(word)
    model.save(path_to_save_mp3)

    audio_clip = AudioFileClip(path_to_save_mp3)
    audio_clip.write_audiofile(path_to_save_wav)
    audio_clip.close()

    print(f"Soubory {mp3_name} a {wav_name} byly uloženy.")


for word in word_list:
    wav_name = f"{word}.wav"
    txt_name = f"{word}.txt"
    path_to_wav = os.path.join(path, wav_name)
    path_to_text = os.path.join(path, txt_name)


    data, samplerate = sf.read(path_to_wav)


    with open(path_to_text, 'w+') as f:
        for val in data:
            f.write(f"{round(10000*float(val[0]))}\n") #zesílení signálu
    print(f"Soubor: {txt_name} byl úspěšně zapsán")


lib_path = './Samples/gtts'
drive_files = [file for file in os.listdir(lib_path) if file.endswith('.txt')]

f = 22050

result_dict = {}
fourier = {}
for item in drive_files:
    file_name = os.path.splitext(item)[0]
    txt_file = f"{file_name}.txt"
    path_to_file = os.path.join(lib_path, txt_file)

    sig = load_signal(path_to_file)
    freq, spec = apply_fourier_transform(sig)

    fourier[item] = freq, spec
    plt.plot(freq, np.abs(spec))
    plt.show()