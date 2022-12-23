# Speech2Plate
Speech2Plate: Generate 3D prints from simple speech.


## What is Speech2Plate?
Speech2Plate is a software tool that allows users to create 3D models for printing, you simply describe the desired object and the 3D Printer will start right away. With Speech2Plate, you can quickly and easily turn your ideas into physical objects. Whether you want to print a custom figurine, a replacement part for a broken appliance, or a unique gift for a friend, Speech2Plate has you covered.

# Workflow

- Speech to text
- text to Point Cloud
- Point Cloud to Mesh
- Mesh to G-Code
- G-Code to 3D printer

# What you need?

- A 3D Printer (Filament and USB Adapter)
- A PC or Laptop with more than 6 GB of VRAM (Working on this to reduce the memory requirements)

# Installation Guide (Windows)

Before beginning, i encourage you to create a new conda environment specifically for this project.

1. First open a new anaconda prompt terminal and run this:

```bash
conda create --name STP python==3.8.15
conda activate STP
```

2. Download this repository and extract it to your desired workflow folder

3. Go to the newly created directory by running:

```bash
cd desktop/Speech2Plate-master
```

4. Install the basic requirements with this command

```bash
pip install -r requirements.txt
```

5. Now install CUDA and CUDNN (Recommended Version: 11.7). You can follow the steps on this guide:
https://medium.com/geekculture/install-cuda-and-cudnn-on-windows-linux-52d1501a8805

6. Now install the Pytorch by running this command:

 ```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```
7. Install Jupyter Notebook by running:

```bash
conda install -c anaconda ipykernel
```
8. To check whether the DL Libraries were installed correctly, open a new jupyter notebook and run this python script:

 ```python
import torch
print("Is CUDA available:",torch.cuda.is_available())
 ```bash
