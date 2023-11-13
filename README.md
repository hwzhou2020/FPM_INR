# FPM_INR
FPM-INR Fourier Ptychographic Microscopy Image Stack Reconstruction using Implicit Neural Representation

The full version of the code has been released.

Project page: https://hwzhou2020.github.io/FPM-INR-Web/

arXiv: https://arxiv.org/abs/2310.18529

Data source: https://doi.org/10.22002/7aer7-qhf77

Top-level folder structure:
```
.
├── data                    # File path for raw / preprocessed FPM data
├── FPM_Matlab              # Matlab code for FPM with first-order optimization (**Parallel computing toolbox** needed)
├── func                    # All-in-focus compuation using LightField method or normal variance method
├── scripts                 # Scripts to run FPM-INR
├── trained_models          # reults save directory
├── vis                     # Result visualization
├── environment.txt         # Anaconda environment
├── FPM_INR.py              # Main python script
├── network.py              # INR nerual nework
├── unils.py                # Utility functions
└── README.md
```


