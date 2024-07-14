#%% Librerías

from utils import plot_image
from sentinelhub import SentinelHubRequest, SentinelHubCatalog, DataCollection, SHConfig, MimeType, MosaickingOrder, geometry
import geopandas as gpd

# %% Logging

config = SHConfig("myprofile")

#%% Geometría

adm_lim_badajoz_raw = gpd.read_file('../../data/Badajoz.geojson').to_crs(epsg=32629)
adm_lim_badajoz = geometry.Geometry(adm_lim_badajoz_raw.geometry[0],adm_lim_badajoz_raw.crs)

#%% Parámetros

time_interval = ("2016-11-30", "2016-12-31")
max_cloud_coverage = 1.0
resolution = (100,100)

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

request_true_color = SentinelHubRequest(
    evalscript = evalscript_true_color,
    input_data = [
        SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L2A.define_from(
                name="s2l2a", service_url="https://sh.dataspace.copernicus.eu"
            ),
            time_interval=time_interval,
            maxcc = max_cloud_coverage,
            mosaicking_order=MosaickingOrder.LEAST_CC
        )
    ],
    responses = [SentinelHubRequest.output_response("default", MimeType.TIFF)],
    geometry = adm_lim_badajoz,
    resolution = resolution,
    config = config,
)

#%% Request
data_full_color = request_true_color.get_data()

#%% Plotting
plot_image(data_full_color[0], factor=3.5 / 255, clip_range=(0, 1))
