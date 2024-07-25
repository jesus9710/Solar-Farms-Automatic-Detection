#%%
from PIL import Image
import os
import rasterio
from rasterio.enums import Resampling
import numpy as np

# Vamos a transformar primero a Tif con las dimensiones del raster original

#%%
# Definir ruta de entrada y salida
input_folder = 'results'
output_image_path = 'reconstructed_preds_image.tif'  

# Listar archivos
files = os.listdir(input_folder)
image_files = [f for f in files if f.endswith('.png')]

# Estas imágenes tienen dos cifras XXXX_XXXX que se corresponden con la posición X/Y en píxeles dentro de la imagen original
# Vamos a usar esto para reconstruir nuestra imagen, tendremos que seleccionar la primera X (0) y recomponer la imagen por columnas
# Para esto extraemos primero las coordenadas
coords = []
for filename in image_files:
    parts = filename.replace('pred_square_', '').replace('.png', '').split('_')
    x, y = int(parts[0]), int(parts[1])
    coords.append((x, y, filename))

# Sacamos los vértices de la imagen original (sólo cuadrados)
if coords:
    min_x = min(x for x, y, _ in coords)
    max_x = max(x for x, y, _ in coords)
    min_y = min(y for x, y, _ in coords)
    max_y = max(y for x, y, _ in coords)

    width = (max_x - min_x) + 200  # ancho
    height = (max_y - min_y) + 200  # alto
else:
    width = height = 0

#%%
# Imagen vacía en escala de grises
reconstructed_image = Image.new('L', (width, height))  # 'L' para escala de grises

# Cargar cada imagen y colocar en su posición
for x, y, filename in coords:
    img_path = os.path.join(input_folder, filename)
    img = Image.open(img_path).convert('L') 

    # Calcular posición
    img_x = x - min_x
    img_y = y - min_y

    # Pegar
    reconstructed_image.paste(img, (img_x, img_y))

# Guardar imagen reconstruida
reconstructed_image.save(output_image_path, format='TIFF')  # Guardar en TIFF para preservar la calidad

#%%
# Vamos a usar el raster original para quemar los valores de las predicciones sobre una de las bandas (RGB)
original_tif_path = 'badajoz0624_prep.tif'
new_tif_path = 'reconstructed_preds_image.tif'
output_tif_path = 'output.tif'

with rasterio.open(original_tif_path) as src:
    # Resolución, extensión y CRS del Tif original
    original_crs = src.crs
    original_transform = src.transform
    original_meta = src.meta.copy()

    original_meta.update({
        'driver': 'GTiff',
        'count': 1,  # Número de bandas, sólo necesitamos una
        'crs': original_crs,
        'transform': original_transform
    })

#%%
with rasterio.open(new_tif_path) as new_src:
    new_image = new_src.read(1, resampling=Resampling.nearest)  # Lee la imagen con la banda 1 y resamplea si es necesario

# Guardar imagen ajustada
with rasterio.open(output_tif_path, 'w', **original_meta) as dst:
    dst.write(new_image, 1)  # Escribe la banda 1

print(f"Imagen ajustada guardada en {output_tif_path}")

#%% Vamos a transformar también a PNG y ajustar los colores para usar en la aplicación

tif_path = 'spatial_data/output.tif'
with rasterio.open(tif_path) as src:
    band = src.read(1) # leer primera banda (única)

# Crear una imagen en blanco con un canal alfa (RGBA)
height, width = band.shape
rgba_image = np.zeros((height, width, 4), dtype=np.uint8)

# Asignar lcolores: 0 = transparencia; 1 = rojo
rgba_image[band == 1] = [255, 0, 0, 255]  # Rojo con opacidad completa
rgba_image[band == 0] = [0, 0, 0, 0]      # Transparente

# Convertir numpy array en una imagen PIL
png_image = Image.fromarray(rgba_image, 'RGBA')

# Guardar la imagen PNG
png_path = 'output.png'
png_image.save(png_path)

print(f"Imagen PNG guardada en {png_path}")
