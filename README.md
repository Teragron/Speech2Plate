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
--------------------------------------------------------------
1. (Optional first Step) If you want to run the script on a Low VRAM Computer, the temporary solution for me was to install the Stable Diffison. You can follow the Optimized SD installation steps from this github: 
https://github.com/basujindal/stable-diffusion

(Note)
If you start with this optimized step you also need to extract and place all the files to the Stable Diffusion folder. During this process, i recommend you not to replace the same named files on the Stable diffusion folder.

-------------------------------------------------------------

2. Download this repository and extract it to your desired workflow folder, and install Git following this tutorial:
https://www.geeksforgeeks.org/how-to-install-git-on-windows-command-line/

3. Go to the newly created directory by running:

```bash
cd desktop/Speech2Plate-main
```

4. Install Jupyter Notebook by running:

```bash
conda install jupyter notebook
```

5. Install the basic requirements with this command

```bash
pip install -r requirements.txt
```

6. Now install CUDA and CUDNN (Recommended Version: 11.7). You can follow the steps on this guide:
https://medium.com/geekculture/install-cuda-and-cudnn-on-windows-linux-52d1501a8805

7. Now install the Pytorch by running this command:

 ```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

8. To check whether the DL Libraries were installed correctly, open a new jupyter notebook and run this python script (This should return True):

 ```python
import torch
print("Is CUDA available:",torch.cuda.is_available())
```
After this is done, you can close the notebook tab.

9. Now we will install three more important libraries, run this command in anaconda terminal:

 ```bash
git clone https://github.com/kliment/Printrun.git
git clone https://github.com/openai/point-e.git
pip install ./point-e
```

10. Download the latest release of Slic3r Software:
https://github.com/slic3r/Slic3r/releases/tag/1.3.0

11. Now create a new folder named "slicer" in your workflow folder and extract everything that you've downloaded from the step 10 to this new "slicer" folder.

12. Now you need a my_config_file.ini file for your 3D Printer, you can get this file by running slic3r.exe in the slicer folder and exporting the configuration for your printer.

13. After getting the my_config_file.ini file, put this inside the main workflow folder and connect your printer to the PC/Laptop and make sure that you are using the right Serial Port. (For me it was COM3)

13. Then simply run this command:
```bash
python tam_speech.py
```

For the low VRAM Version, run this command
```bash
python optimized_tam.py
```
-----------------------------------------------------
Now you have 5 second to describe the desired Object.
-----------------------------------------------------
The first time you run all these commands, the script will start to download all necessary model files. It may take a while.
Even if you don't plug the device to your 3D Printer, the script still going to create a .stl file that can be transferred via SD-Card. A Downside of using USB- Port is that the data transfer can be easily interrupted and the Port Connection is weak for processing .gcode data. If you have a Klipper or Oktopi Connection method that i can implement to this script via Raspberry Pi, i would be very happ to add that to this repository.

# Demo

https://user-images.githubusercontent.com/54573938/209484504-bfefaa3a-cb62-45d7-88eb-53ef60cc9b8d.mp4



# Plans

I am planning to add a Dreambooth model to the script so that the VRAM requirements can be reduced and the fine-tuned images can have a style which is more compatible with the Image2PointCloud script of Point-E. (In style of corgi.jpg or cube_stack.jpg from point-e GitHub)

# Disclaimer

Be careful when using the script, the configuration file that i have provided is only for my 3D-Printer and might won't work for yours. The Bed and the Nozzle Temp. are both critical parameters when using the 3D-Printer and need to be set via the configuration.ini file. I take no responsibility for any damage caused by the user due to misuse of this script or any related prerequisites.
