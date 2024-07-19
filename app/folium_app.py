#%%
import geopandas as gpd
import folium
import rasterio
from folium.raster_layers import ImageOverlay
from matplotlib import cm
import numpy as np

#%%
# Cargar archivos vectoriales
shp1 = gpd.read_file('spatial_data/Es_Lic_SCI_Zepa_SPA_Medalpatl_202401.shp')
shp2 = gpd.read_file('spatial_data/FV_sensibilidad_vector.shp')
geojson = gpd.read_file('spatial_data/Badajoz.geojson')

# %%
import folium
from folium import IFrame
from folium.plugins import Geocoder

# Coordenadas centrales aproximadas (ajústalas según sea necesario)
center_lat, center_lon = 39.4702, -6.3722

# URL del WMS
wms_url = "https://wms.mapama.gob.es/sig/Biodiversidad/RedNatura/wms.aspx"

# URL de la leyenda
legend_url = "https://wms.mapama.gob.es/sig/Biodiversidad/RedNatura/Leyenda/RedNatura.png"

# Crear el mapa
m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

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
m.save('mapa_wms_leyenda2.html')

m

# %%
