#%% Visualización de la provincia

import geopandas as gpd
import matplotlib.pyplot as plt

# Cargar el archivo GeoJSON
geojson_file = '../../data/Badajoz.geojson'
gdf = gpd.read_file(geojson_file)

# Verificar el contenido del GeoDataFrame
print(gdf.head())

# Si necesitas filtrar por un campo específico, asegúrate de conocer el nombre del campo y el valor a filtrar
# Por ejemplo, si el campo es 'NAME_2' y el valor es 'Badajoz'
# gdf_badajoz = gdf[gdf['NAME_2'] == 'Badajoz']

# Visualizar el polígono
gdf.plot()
plt.show()

# %%
