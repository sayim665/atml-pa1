"""
Paste this into a Colab cell to run Task 2's source_only.py after cloning your repo
and adding the shared/ and models/ folders to sys.path. See task1's notebook pattern
for the Drive-mount + pip-install boilerplate -- reuse that here first.

!pip install -q datasets torchvision scikit-learn
!git clone https://github.com/sayim665/atml-pa1.git /content/atml-pa1   # if not already cloned

import sys
sys.path.insert(0, "/content/atml-pa1/task2/shared")
sys.path.insert(0, "/content/atml-pa1/task2/models")
sys.path.insert(0, "/content/atml-pa1/task2/methods")

from source_only import main
results = main(
    results_dir="/content/drive/MyDrive/atml_pa1_task2/results",
    checkpoint_dir="/content/drive/MyDrive/atml_pa1_task2/checkpoints",
)
"""
