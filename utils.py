import numpy as np
import matplotlib as mpl
from matplotlib.colors import ListedColormap

newcolors = np.vstack(
    (
        np.flipud(mpl.colormaps['magma'](np.linspace(0, 1, 128))),
        mpl.colormaps['magma'](np.linspace(0, 1, 128)),
    )
)
newcmp = ListedColormap(newcolors, name='magma_cyclic')

