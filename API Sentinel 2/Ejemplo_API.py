#%% Librerías

import requests
import pandas as pd
from functions import *

#%% Credenciales

creds = get_credentials('creds.txt')

#%% Token de acceso

access_token = get_access_token(creds.user, creds.password)

# %% Área de interés

start_date = "2022-06-01"
end_date = "2022-06-02"
data_collection = "SENTINEL-2"
aoi = "POLYGON((4.220581 50.958859,4.521264 50.953236,4.545977 50.906064,4.541858 50.802029,4.489685 50.763825,4.23843 50.767734,4.192435 50.806369,4.189689 50.907363,4.220581 50.958859))'"

# %% Requests

json = requests.get(f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=Collection/Name eq '{data_collection}' and OData.CSC.Intersects(area=geography'SRID=4326;{aoi}) and ContentDate/Start gt {start_date}T00:00:00.000Z and ContentDate/Start lt {end_date}T00:00:00.000Z").json()

# %% Dataframe

json_df = pd.DataFrame.from_dict(json['value'])
products = json_df.Id.to_list()

# %% Descarga de un producto

c_Id = products[0]
c_filename = "product1"

# download_product(access_token = access_token, Id = c_Id, filename = c_filename)

# %% Descarga de varios productos

Ids = [products[0] ,products[1]]

dir = 'C:/Jesús/5_Programacion/Local Scripts Python/TFM Kschool/Instalaciones fotovoltaicas/Token Creation/Prueba_2'

download_products(access_token = access_token, Ids = Ids, dir=dir)

# %%
