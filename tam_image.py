import os
import time
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

image_name = "corgi" #without the extension
print(f"Working with {image_name}")

def extract_first_chars(sentence):
    words = sentence.split()
    first_chars = [word[0] for word in words]
    return ''.join(first_chars)

filename = image_name


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
base_name = 'base40M' # use base300M or base1B for better results
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
    guidance_scale=[3.0, 3.0],
)

print("Is CUDA available:",torch.cuda.is_available())


# Load an image to condition on.
try:
    img = Image.open(f'./point-e/point_e/examples/example_data/{image_name}.jpg')
except:
    img = Image.open(f'./point-e/point_e/examples/example_data/{image_name}.png)

dim = (256, 256)
width, height = img.size

if (width, height) != dim:
    img = img.resize(dim)
    img = img.convert('RGB')
    img.save(f'./point-e/point_e/examples/example_data/{image_name}.jpg')

samples = None
for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(images=[img]))):
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
