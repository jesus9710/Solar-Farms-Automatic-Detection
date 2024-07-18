#%% Librerías

from utils import show_splitter, get_sh_request_from_split, get_product_path, merge_products
from sentinelhub import SentinelHubRequest, SentinelHubCatalog, SentinelHubDownloadClient, DataCollection, SHConfig, MimeType, MosaickingOrder, geometry, BBoxSplitter
import geopandas as gpd
from shapely.geometry import box

# %% Logging

config = SHConfig("myprofile")

#%% Geometría

lim_raw = gpd.read_file('../../data/recintos_provinciales_inspire_peninbal_etrs89.shp').to_crs(epsg=32630) # Cambio a EPSG32630 (WGS 84 / UTM zone 30N) https://epsg.io/32630
lim = lim_raw[lim_raw['NAMEUNIT'].isin(['Badajoz', 'Ciudad Real'])]
polygon = lim.unary_union

# bounding box
minx, miny, maxx, maxy = polygon.bounds
bounding_box = box(minx, miny, maxx, maxy)
# bbox as geodataframe
bbox_gdf = gpd.GeoDataFrame(index=[0], crs=lim.crs, geometry=[bounding_box])
bbox_geometry = geometry.Geometry(bbox_gdf.geometry[0],bbox_gdf.crs) # geometry
bbox_split = BBoxSplitter(bbox_gdf.geometry, bbox_gdf.crs,(30,30)) # bbox splitted to download
# save gdf for next script
#bbox_gdf.to_file('../../data/bbox_Badajoz_CReal.geojson', driver='GeoJSON')


#%% Comprobar box splitter

adm_lim_badajoz_raw_4326 = gpd.read_file('../../data/Badajoz.geojson').to_crs(epsg=4326) # La función show_splitter no admite EPSG 32629, por lo que se hace el cambio a EPSG4326
adm_lim_badajoz_split_4326 = BBoxSplitter(adm_lim_badajoz_raw_4326.geometry, adm_lim_badajoz_raw_4326.crs,(15,15),reduce_bbox_sizes=True)
show_splitter(adm_lim_badajoz_split_4326) 

#%% Parámetros

time_interval = ("2019-10-01", "2024-12-31")
max_cloud_coverage = 1.0
resolution= (10,10)

#%% CATALOG API

catalog = SentinelHubCatalog(config=config)

search_iterator = catalog.search(
    DataCollection.SENTINEL2_L2A,
    geometry=bbox_geometry,
    time=time_interval,
    fields={"include": ["id", "properties.datetime"], "exclude": []},
    filter='eo:cloud_cover < ' + str(max_cloud_coverage*100),
)

results = list(search_iterator)
print("Total number of results:", len(results))

results

#%% PROCESS API

evalscript_true_color = """
    //VERSION=3

    function setup() {
        return {
            input: [{
                bands: ["B02", "B03", "B04"]
            }],
            output: {
                bands: 3
            }
        };
    }

    function evaluatePixel(sample) {
        return [sample.B04, sample.B03, sample.B02];
    }
"""

input_data=[
    SentinelHubRequest.input_data(
        data_collection=DataCollection.SENTINEL2_L2A.define_from(
            name="s2l2a", service_url="https://sh.dataspace.copernicus.eu"
        ),
        time_interval=time_interval,
        maxcc = max_cloud_coverage,
        mosaicking_order=MosaickingOrder.LEAST_CC
    )
]
responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)]

sh_requests, dl_requests = get_sh_request_from_split(bbox_split,
                                                    evalscript=evalscript_true_color,
                                                    input_data=input_data,
                                                    responses=responses,
                                                    resolution=resolution,
                                                    config=config)

#%% Descarga

data_full_color = SentinelHubDownloadClient(config=config).download(dl_requests, max_threads=None)

#%% Merge

file_name = 'merged_raster.tiff'
save_file = True # Flag de seguridad para evitar sobreescribir imágenes

if save_file:
    tiffs = get_product_path(sh_requests)
    merge_products(tiffs, file_name)
