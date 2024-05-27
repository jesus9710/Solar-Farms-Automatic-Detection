#%% Librerías

from utils import plot_image, show_splitter, get_sh_request_from_split, get_product_path, merge_products
from sentinelhub import SentinelHubRequest, SentinelHubCatalog, SentinelHubDownloadClient, DataCollection, SHConfig, MimeType, MosaickingOrder, geometry, BBoxSplitter
import geopandas as gpd

# %% Logging

config = SHConfig("myprofile")

#%% Geometría

adm_lim_badajoz_raw = gpd.read_file('../../data/Badajoz.geojson').to_crs(epsg=32629) # Cambio a EPSG32629 (WGS 84 / UTM zone 29N) https://epsg.io/32629
adm_lim_badajoz = geometry.Geometry(adm_lim_badajoz_raw.geometry[0],adm_lim_badajoz_raw.crs) # Geometría de badajoz
adm_lim_badajoz_split = BBoxSplitter(adm_lim_badajoz_raw.geometry, adm_lim_badajoz_raw.crs,(15,15)) # Geometría separada en diferentes BBox

#%% Comprobar box splitter

adm_lim_badajoz_raw_4326 = gpd.read_file('../../data/Badajoz.geojson').to_crs(epsg=4326) # La función show_splitter no admite EPSG 32629, por lo que se hace el cambio a EPSG4326
adm_lim_badajoz_split_4326 = BBoxSplitter(adm_lim_badajoz_raw_4326.geometry, adm_lim_badajoz_raw_4326.crs,(15,15),reduce_bbox_sizes=True)
show_splitter(adm_lim_badajoz_split_4326) 

#%% Parámetros

time_interval = ("2016-11-30", "2016-12-31")
max_cloud_coverage = 1.0
resolution= (10,10)

#%% CATALOG API

catalog = SentinelHubCatalog(config=config)

search_iterator = catalog.search(
    DataCollection.SENTINEL2_L2A,
    geometry=adm_lim_badajoz,
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

sh_requests, dl_requests = get_sh_request_from_split(adm_lim_badajoz_split,
                                                    evalscript=evalscript_true_color,
                                                    input_data=input_data,
                                                    responses=responses,
                                                    resolution=resolution,
                                                    config=config)

'''
evalscript=evalscript,
        input_data=input_data,
        responses=responses,
        bbox=bbox,
        resolution=resolution,
        data_folder=tempfile.gettempdir(),
        config=config,
'''
#%% Obtener un preview

get_preview = False

if get_preview:
    data_full_color = []
    for request in sh_requests:
        data_full_color.append(request.get_data()[0])
    for image in data_full_color:
        plot_image(image, factor=3.5 / 255, clip_range=(0, 1))

#%% Descarga

data_full_color = SentinelHubDownloadClient(config=config).download(dl_requests, max_threads=None)

#%% Merge

file_name = 'merged_rasted.tiff'
save_file = True # Flag de seguridad para evitar sobreescribir imágenes

if save_file:
    tiffs = get_product_path(sh_requests)
    merge_products(tiffs, file_name)

#%% Plotting


# %%
