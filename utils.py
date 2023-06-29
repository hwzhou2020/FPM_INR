import numpy as np
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import torch

def save_model_with_required_grad(model, save_path):
    tensors_to_save = []
    
    # Traverse through model parameters and append tensors with require_grad=True to the list
    for param_name, param_tensor in model.named_parameters():
        if param_tensor.requires_grad:
            tensors_to_save.append(param_tensor)
    
    # Save the list of tensors
    torch.save(tensors_to_save, save_path)

newcolors = np.vstack(
    (
        np.flipud(mpl.colormaps['magma'](np.linspace(0, 1, 128))),
        mpl.colormaps['magma'](np.linspace(0, 1, 128)),
    )
)
newcmp = ListedColormap(newcolors, name='magma_cyclic')

