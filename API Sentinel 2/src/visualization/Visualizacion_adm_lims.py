#%% Visualización de la provincia

import geopandas as gpd
import matplotlib.pyplot as plt

# Cargar el archivo GeoJSON
geojson_file = '../../data/Badajoz.geojson'
gdf = gpd.read_file(geojson_file)

# Verificar el contenido del GeoDataFrame
print(gdf.head())

# Visualizar el polígono
gdf.plot()
plt.show()

# %%
