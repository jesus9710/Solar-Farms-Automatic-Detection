#%% Libs
import numpy as np
from matplotlib import pyplot as plt

from shapely.geometry import shape, MultiPolygon, Polygon
import rasterio
from rasterio.mask import mask as rasterio_mask
from rasterio.plot import show

from pathlib import Path

# %% Abrir archivo

# rutas
directorio_datos = Path.cwd() / 'spatial_data'
multiclass_mask_filename = "badajoz_multiclass_mask.tiff"
rgb_filename = 'merged_raster_preprocessed.tiff'
mask_path = directorio_datos / multiclass_mask_filename
rgb_path = directorio_datos / rgb_filename

# Leer la imagen RGB
with rasterio.open(rgb_path) as src_rgb:
    rgb_image = src_rgb.read([1, 2, 3])  # Leer las tres bandas de la imagen RGB
    profile = src_rgb.profile

# Leer la banda donde están todas las máscaras
with rasterio.open(mask_path) as src_mask:
    band = src_mask.read(1) # parece que todas las máscaras están en una sola banda

# %% Obtención de máscaras

mascaras = []
for value in np.unique(band):
    if value != 0:
        # si band = value -> asignamos un 1, si no, un 0
        mascara = np.where(band == value, 1, 0).astype(np.uint8)
        mascaras.append(mascara)

# Las máscaras están en la lista máscaras. len(mascaras) = 5

#%% graficar máscaras

# Aquí se pueden graficar las máscaras

# show(mascaras[1])

# Aquí se pueden graficar las máscaras aplicadas sobre la imagen original:

masked_rgb = np.where(mascaras[1] == 1, rgb_image, 0)

show(masked_rgb)

# %% Geometrías

# Aquí se intenta extraer la geometría de las máscaras

Polygon_value_generator = rasterio.features.shapes(mascaras[1], transform=src_mask.transform) # con transform=src_mask.transform se conserva el mismo sistema de referencia
geometries = [shape(geom) for geom, _ in Polygon_value_generator]

with rasterio.open(rgb_path) as src_rgb:
    masked_image, _ = rasterio_mask(src_rgb, [geometries[1]] , crop = True)


# %% Graficar imagenes recortada por geometría

show(masked_image)

'''
Parece que no estña funcionando bien. Creo que la variable geometries debe ser del tipo:

<POLYGON (825931.36 4356650.345, 825931.36 4356630.349, 826021.341 4356630....

En lugar de:

<POLYGON ((825931.36 4356650.345, 825931.36 4356630.349, 826021.341 4356630....

De todas formas, Laura tiene las geometrías.

'''

# %%

print('prueba')
