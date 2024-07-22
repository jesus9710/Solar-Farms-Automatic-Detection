#%%
import geopandas as gpd
from pyproj import Transformer
import folium
import rasterio
from folium.raster_layers import ImageOverlay
from matplotlib import cm
import numpy as np
from folium import IFrame
from folium.plugins import Geocoder

# Ruta al Tif y al PNG
# PNG para visualizar y Tif para establecer límites
tif_file = 'spatial_data/output.tif'
png_file = 'spatial_data/output.png'

with rasterio.open(tif_file) as src:
    bounds = src.bounds
    crs = src.crs

# Transformador a EPSG:4326 (porque los mapas de Folium se basan en ese CRS)
transformer = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)

# Transformar los límites
left, bottom = transformer.transform(bounds.left, bounds.bottom)
right, top = transformer.transform(bounds.right, bounds.top)

# Crear el mapa base centrado en la ubicación del raster
m = folium.Map(location=[(bottom + top) / 2, (left + right) / 2], zoom_start=10)

# Definir los límites geográficos de la imagen PNG
image_bounds = [[bottom, left], [top, right]]

# Añadir imagen de predicciones del modelo al mapa base
overlay = ImageOverlay(name='Paneles solares (IA)',
                       image=png_file,
                       bounds=image_bounds,
                       opacity=1)
overlay.add_to(m)

# URL del WMS
wms_url = "https://wms.mapama.gob.es/sig/Biodiversidad/RedNatura/wms.aspx"

# URL de la leyenda
legend_url = "https://wms.mapama.gob.es/sig/Biodiversidad/RedNatura/Leyenda/RedNatura.png"

# Añadir capa de OpenStreetMap (callejero)
folium.TileLayer('openstreetmap').add_to(m)

# Añadir capa de Google Satellite
google_satellite = folium.TileLayer(
    tiles='http://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
    attr='Google',
    name='Google Satellite',
    overlay=True,
    control=True
)
google_satellite.add_to(m)

# Agregar servidor WMS
folium.WmsTileLayer(
    url=wms_url,
    name='Red Natura',
    layers='Red Natura 2000',  # 'title' de la capa de interés
    format='image/png',
    transparent=True,
    control=True,
    overlay=True,
    opacity=0.3 # transparencia
).add_to(m)

# Crear un marco HTML para la leyenda
legend_html = f'''
     <div style="
     position: fixed; 
     bottom: 20px; right: 10px; 
     z-index: 9999; 
     ">
     <img src="{legend_url}" alt="Legend">
     </div>
     '''

# Agregar el HTML de la leyenda al mapa
m.get_root().html.add_child(folium.Element(legend_html))

folium.WmsTileLayer(
    url="http://6ae9b70.online-server.cloud/geoserver/sensibilidad_renovables/wms",
    name='Sensibilidad Fotovoltaicas',
    layers='Sensibilidad_fotovoltaica',  # Asegúrate de utilizar el nombre correcto de la capa
    format='image/png',
    transparent=True,
    control=True,
    overlay=True,
    opacity=0.5  # transparencia
).add_to(m)

legend_url_seo = 'https://lh3.googleusercontent.com/u/0/drive-viewer/AKGpiha92BY-wIJ9u_tiXqDCM8hIYK6QQuNGLFiGtNnEw-27tCvOW24nl1XvAu6DcNf9jHoIh4M7L-fjdsdwinZJ9Z2kxaNprkAeduY=w1920-h888'

legend_seo = f'''
     <div style="
     position: fixed; 
     bottom: 110px; right: 10px; 
     z-index: 9999; 
     ">
     <img src="{legend_url_seo}" alt="Legend" width="80" height="160">
     </div>
     '''
m.get_root().html.add_child(folium.Element(legend_seo))

# Barra de búsqueda
Geocoder().add_to(m)

# Control de capas
folium.LayerControl().add_to(m)

# Guardar el mapa en un archivo HTML
m.save('app_valoracion_fotovoltaicas.html')

m

# %%
