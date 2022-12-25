import os
import time
from PIL import Image

import torch
import whisper

import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wvimport, subprocess



freq = 88200
duration = 5
recording = sd.rec(int(duration * freq), samplerate=freq, channels=2)
print("Recording is started...")
sd.wait()
write("recording0.wav", freq, recording)


model = whisper.load_model("medium")
audio = whisper.load_audio("recording0.wav")
audio = whisper.pad_or_trim(audio)
mel = whisper.log_mel_spectrogram(audio).to(model.device)
_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")

options = whisper.DecodingOptions()
result = whisper.decode(model, mel, options)

def extract_first_chars(sentence):
    words = sentence.split()
    first_chars = [word[0] for word in words]
    return ''.join(first_chars)

filename = extract_first_chars(str(result.text))
print(result.text)


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

my_prompt = str(result.text)
print(f"Working with {my_prompt}")

torch.cuda.empty_cache()

def extract_first_chars(sentence):
    words = sentence.split()
    first_chars = [word[0] for word in words]
    return ''.join(first_chars)


import os
import time
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def extract_first_chars(sentence):
    words = sentence.split()
    first_chars = [word[0] for word in words]
    return ''.join(first_chars)

def replace_spaces(s):
    return s.replace(" ", "_")


filename = my_prompt

patho = replace_spaces(f'a 3d model of {my_prompt} isometric view white background')

with open('okubeni.txt', 'w') as f:
    f.write(f'{patho}')
command_first = ['python', 'optimizedSD/optimized_txt2img.py', '--prompt', f'a 3d model of {my_prompt} isometric view white background', '--H', '512', '--W', '512', '--n_iter', '1', '--n_samples', '1', '--sampler', 'euler_a', '--ddim_steps', '30', '--skip_grid', '--turbo', '--precision', 'full', "--outdir", "bura/", "--seed", "5"]
subprocess.run(command_first)