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




from PIL import Image
import torch
from tqdm.auto import tqdm

from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.plotting import plot_point_cloud

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('creating base model...')
base_name = 'base40M-textvec'
base_model = model_from_config(MODEL_CONFIGS[base_name], device)
base_model.eval()
base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

print('creating upsample model...')
upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
upsampler_model.eval()
upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])

print('downloading base checkpoint...')
base_model.load_state_dict(load_checkpoint(base_name, device))

print('downloading upsampler checkpoint...')
upsampler_model.load_state_dict(load_checkpoint('upsample', device))

sampler = PointCloudSampler(
    device=device,
    models=[base_model, upsampler_model],
    diffusions=[base_diffusion, upsampler_diffusion],
    num_points=[1024, 4096 - 1024],
    aux_channels=['R', 'G', 'B'],
    guidance_scale=[3.0, 0.0],
    model_kwargs_key_filter=('texts', ''), # Do not condition the upsampler at all
)

print("Is CUDA available:",torch.cuda.is_available())

# Set a prompt to condition on.
prompt = 'a red motorcycle'

# Produce a sample from the model.
samples = None
for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(texts=[prompt]))):
    samples = x

pc = sampler.output_to_point_clouds(samples)[0]
# fig = plot_point_cloud(pc, grid_size=3, fixed_bounds=((-0.75, -0.75, -0.75),(0.75, 0.75, 0.75)))
pc.save(f'{filename}.npz')


from PIL import Image
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.pc_to_mesh import marching_cubes_mesh
from point_e.util.plotting import plot_point_cloud
from point_e.util.point_cloud import PointCloud

import skimage.measure

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('creating SDF model...')
name = 'sdf'
model = model_from_config(MODEL_CONFIGS[name], device)
model.eval()

print('loading SDF model...')
model.load_state_dict(load_checkpoint(name, device))

# # Load a point cloud we want to convert into a mesh.
pc = PointCloud.load(f'{filename}.npz')


# Produce a mesh (with vertex colors)
mesh = marching_cubes_mesh(
    pc=pc,
    model=model,
    batch_size=4096,
    grid_size=32, # increase to 128 for resolution used in evals
    progress=True,
)

# Write the mesh to a PLY file to import into some other program.
with open(f'{filename}.ply', 'wb') as f:
    mesh.write_ply(f)
    
    
    
import subprocess
import aspose.threed as a3d
from printrun.printcore import printcore
from printrun import gcoder


scene = a3d.Scene.from_file(f"{filename}.ply")
scene.save(f"{filename}.stl")


subprocess.run(["slicer/slic3r-console", "--load", "my_config_file.ini", f"{filename}.stl"])
try:
    p = printcore('COM3',115200) 
except:
    print("By default, the 3D Printer is plugged on COM3 USB Port, you can try change the Serial from COM3 to COM5")

gcode=[i.strip() for i in open(f'{filename}.gcode')] 
gcode = gcoder.LightGCode(gcode)

while not p.online:
    time.sleep(0.1)

p.startprint(gcode)