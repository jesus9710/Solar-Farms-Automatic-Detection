#%% Librerías:

import numpy as np
import rasterio
import geopandas as gpd

from pathlib import Path

# %% Abrir archivos

file_name = "merged_raster.tiff"

image_path = Path.cwd().parent.parent / ('data/' + file_name)
tiff_image = rasterio.open(image_path)

adm_lim_badajoz = gpd.read_file('../../data/Badajoz.geojson').to_crs(epsg=32629)

#%% preprocesado

brightness_factor = 3.5

new_file_name = "merged_raster_preprocessed"
new_image_path = Path.cwd().parent.parent / ('data/' + new_file_name)

save_file = False # Flag de seguridad

if save_file:
    with rasterio.open(image_path) as src:

        # Recortar la imagen usando las geometrías
        out_image, out_transform = rasterio.mask.mask(src, adm_lim_badajoz.geometry, crop=True)
        out_meta = src.meta.copy()

        # Aumentar el brillo
        out_image = np.clip(out_image * brightness_factor, 0, 255)

        # Actualizar metadatos para el archivo recortado
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        # Guardar la imagen recortada y con el brillo ajustado en formato TIFF
        with rasterio.open(new_image_path, 'w', **out_meta) as dest:
            dest.write(out_image.astype(out_meta['dtype']))
