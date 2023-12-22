# FPM_INR
FPM-INR Fourier Ptychographic Microscopy Image Stack Reconstruction using Implicit Neural Representation

The full version of the code has been released.

Paper link: https://doi.org/10.1364/OPTICA.505283

Project page: https://hwzhou2020.github.io/FPM-INR-Web/

arXiv: https://arxiv.org/abs/2310.18529

Data source: https://doi.org/10.22002/7aer7-qhf77

Top-level folder structure:
```
.
├── data                    # File path for raw / preprocessed FPM data
├── FPM_Matlab              # Matlab code for FPM with first-order optimization (Parallel computing toolbox needed)
├── func                    # All-in-focus computation using LightField method or normal variance method
├── scripts                 # Scripts to run FPM-INR
├── trained_models          # reults save directory
├── vis                     # Result visualization
├── environment.txt         # Anaconda environment
├── FPM_INR.py              # Main Python script
├── network.py              # INR neural network
├── unils.py                # Utility functions
└── README.md
```

## BiBTeX

```
@article{Zhou2023fpminr,
  author = {Haowen Zhou and Brandon Y. Feng and Haiyun Guo and Siyu (Steven) Lin and Mingshu Liang and Christopher A. Metzler and Changhuei Yang},
  journal = {Optica},
  keywords = {Biomedical imaging; Computer simulation; Deep learning; Neural networks; Phase retrieval; Systems design},
  number = {12},
  pages = {1679--1687},
  publisher = {Optica Publishing Group},
  title = {Fourier ptychographic microscopy image stack reconstruction using implicit neural representations},
  volume = {10},
  month = {Dec},
  year = {2023},
  url = {https://opg.optica.org/optica/abstract.cfm?URI=optica-10-12-1679},
  doi = {10.1364/OPTICA.505283}
}
```

