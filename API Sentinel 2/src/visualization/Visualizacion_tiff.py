# %% Librerías

from utils import plot_image
from matplotlib import pyplot as plt

import rioxarray
import rasterio
from rasterio.plot import show

import os
from pathlib import Path

# %% Abrir archivo

file_name = "merged_rasted.tiff"

image_path = Path(os.getcwd().split('src')[0]+'\\data\\'+file_name)
tiff_image = rasterio.open(image_path)
xarray_image = rioxarray.open_rasterio(image_path)

# %% Representación gráfica

fig, ax = plt.subplots(figsize=(30,30))
show(tiff_image, ax=ax)
plot_image(xarray_image.values.transpose(1,2,0), factor=3.5 / 255, clip_range=(0, 1))

# %%
