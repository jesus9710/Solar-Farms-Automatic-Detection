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

#%%
# Crear el mapa
lon, lat = geojson.geometry.centroid.iloc[0].x, geojson.geometry.centroid.iloc[0].y
m = folium.Map(zoom_start=12)

# Agregar capas vectoriales
folium.GeoJson(shp1).add_to(m)
folium.GeoJson(shp2).add_to(m)
folium.GeoJson(geojson).add_to(m)

#%%
# Cargar y agregar el archivo raster
with rasterio.open('path_to_raster.tif') as src:
    raster_array = src.read(1)
    bounds = src.bounds

norm_array = (raster_array - np.min(raster_array)) / (np.max(raster_array) - np.min(raster_array))
cmap = cm.get_cmap('viridis')
rgba_array = cmap(norm_array)
rgba_array = (rgba_array * 255).astype(np.uint8)

img_overlay = ImageOverlay(
    image=rgba_array,
    bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
    opacity=0.6,
    interactive=True
)

img_overlay.add_to(m)

#%%
# Guardar el mapa como archivo HTML
m.save('map.html')

# %%
import folium

# Coordenadas centrales aproximadas (ajústalas según sea necesario)
center_lat, center_lon = 39.4702, -6.3722

# URL del WMS
wms_url = "https://wms.mapama.gob.es/sig/Biodiversidad/RedNatura/wms.aspx"

# Crear el mapa
m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

# Agregar capas de OpenStreetMap
folium.TileLayer('openstreetmap').add_to(m)

# Agregar servidor WMS con transparencia
folium.WmsTileLayer(
    url=wms_url,
    name='Red Natura',
    layers='Red Natura 2000',  # Asegúrate de utilizar el nombre correcto de la capa
    format='image/png',
    transparent=True,
    control=True,
    overlay=True,
    opacity=0.5  # Ajusta este valor entre 0 (completamente transparente) y 1 (completamente opaco) según tus necesidades
).add_to(m)

# Agregar control de capas
folium.LayerControl().add_to(m)

# Guardar el mapa en un archivo HTML
m.save('mapa_wms.html')

m



# %%
import folium
from folium import IFrame

# Coordenadas centrales aproximadas (ajústalas según sea necesario)
center_lat, center_lon = 39.4702, -6.3722

# URL del WMS
wms_url = "https://wms.mapama.gob.es/sig/Biodiversidad/RedNatura/wms.aspx"

# URL de la leyenda
legend_url = "https://wms.mapama.gob.es/sig/Biodiversidad/RedNatura/Leyenda/RedNatura.png"

# Crear el mapa
m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

# Agregar capas de OpenStreetMap
folium.TileLayer('openstreetmap').add_to(m)

# Agregar servidor WMS con transparencia
folium.WmsTileLayer(
    url=wms_url,
    name='Red Natura',
    layers='Red Natura 2000',  # Asegúrate de utilizar el nombre correcto de la capa
    format='image/png',
    transparent=True,
    control=True,
    overlay=True,
    opacity=0.3  # Ajusta este valor entre 0 (completamente transparente) y 1 (completamente opaco) según tus necesidades
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
    opacity=0.5  # Ajusta este valor entre 0 (completamente transparente) y 1 (completamente opaco) según tus necesidades
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

# Agregar control de capas
folium.LayerControl().add_to(m)

# Guardar el mapa en un archivo HTML
m.save('mapa_wms_leyenda.html')

m

# %%
import folium

# Coordenadas centrales aproximadas (ajústalas según sea necesario)
center_lat, center_lon = 39.4702, -6.3722

# URL del WMS
wms_url = "https://wms.mapama.gob.es/sig/Biodiversidad/RedNatura/wms.aspx"

# URL de la leyenda
legend_url = "https://wms.mapama.gob.es/sig/Biodiversidad/RedNatura/Leyenda/RedNatura.png"

# Crear el mapa
m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

# Agregar capas de OpenStreetMap
folium.TileLayer('openstreetmap').add_to(m)

# Agregar servidor WMS con transparencia
folium.WmsTileLayer(
    url=wms_url,
    name='Red Natura',
    layers='Red Natura 2000',  # Asegúrate de utilizar el nombre correcto de la capa
    format='image/png',
    transparent=True,
    control=True,
    overlay=True,
    opacity=0.3  # Ajusta este valor entre 0 (completamente transparente) y 1 (completamente opaco) según tus necesidades
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
    opacity=0.5  # Ajusta este valor entre 0 (completamente transparente) y 1 (completamente opaco) según tus necesidades
).add_to(m)

legend_url_seo = 'https://lh3.googleusercontent.com/u/0/drive-viewer/AKGpihYzjen0hjFqyeRSvaha_C9ssSlb5k_NENtJc_hgjCcpceh75vwkr1wcFuuLS3qumYjNVflJ5TxrwjYqGxp1rjdpuKvXNl-nZqM=w1920-h888'

legend_seo = f'''
     <div style="
     position: fixed; 
     bottom: 110px; right: 10px; 
     z-index: 9999; 
     ">
     <img src="{legend_url_seo}" alt="Legend">
     </div>
     '''
m.get_root().html.add_child(folium.Element(legend_seo))

# Código HTML y JavaScript para incluir el plugin Leaflet Control Geocoder
geocoder_plugin = """
<!DOCTYPE html>
<html>
<head>
    <title>Mapa con Búsqueda</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
    <link rel="stylesheet" href="https://unpkg.com/leaflet-control-geocoder/dist/Control.Geocoder.css" />
    <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
    <script src="https://unpkg.com/leaflet-control-geocoder/dist/Control.Geocoder.js"></script>
</head>
<body>
    <div id="map" style="width: 100%; height: 600px;"></div>
    <script>
        var map = L.map('map').setView([39.4702, -6.3722], 12);
        
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors'
        }).addTo(map);
        
        L.Control.geocoder().addTo(map);

        // Agregar las capas WMS
        L.tileLayer.wms('https://wms.mapama.gob.es/sig/Biodiversidad/RedNatura/wms.aspx', {
            layers: 'Red Natura 2000',
            format: 'image/png',
            transparent: true,
            opacity: 0.3
        }).addTo(map);

        L.tileLayer.wms('http://6ae9b70.online-server.cloud/geoserver/sensibilidad_renovables/wms', {
            layers: 'Sensibilidad_fotovoltaica',
            format: 'image/png',
            transparent: true,
            opacity: 0.5
        }).addTo(map);

        // Agregar las leyendas
        var legend1 = L.control({position: 'bottomright'});
        legend1.onAdd = function (map) {
            var div = L.DomUtil.create('div', 'info legend');
            div.innerHTML += '<img src="https://wms.mapama.gob.es/sig/Biodiversidad/RedNatura/Leyenda/RedNatura.png" alt="Legend" />';
            return div;
        };
        legend1.addTo(map);

        var legend2 = L.control({position: 'bottomright'});
        legend2.onAdd = function (map) {
            var div = L.DomUtil.create('div', 'info legend');
            div.innerHTML += '<img src="http://6ae9b70.online-server.cloud/geoserver/sensibilidad_renovables/ows?service=WMS&version=1.3.0&request=GetLegendGraphic&format=image%2Fpng&width=20&height=20&layer=Sensibilidad_fotovoltaica" alt="Legend" />';
            return div;
        };
        legend2.addTo(map);
    </script>
</body>
</html>
"""

# Agregar control de capas
folium.LayerControl().add_to(m)

# Guardar el código en un archivo HTML
with open('mapa_con_busqueda.html', 'w') as f:
    f.write(geocoder_plugin)

m

# %%
