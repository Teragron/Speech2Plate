import os
import subprocess
import torch

# Run the first script
subprocess.run(["python", "tam_image_first.py"])

# Close the terminal window
os.system("exit")

torch.cuda.empty_cache()

# Run the second script
subprocess.run(["python", "tam_image_second.py"])

