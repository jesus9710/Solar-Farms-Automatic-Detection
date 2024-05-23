#%% Librerías

import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
from sentinelhub import (CRS,
                         BBox,
                         DataCollection,
                         SHConfig,
                         DownloadRequest,
                         MimeType,
                         MosaickingOrder,
                         SentinelHubDownloadClient,
                         SentinelHubRequest,
                         bbox_to_dimensions,
)

from sentinelhub import SentinelHubCatalog
import os
from utils import plot_image

# %% Logging

config = SHConfig("myprofile")

#%% Parámetros

aoi_coords_wgs84 = (-6.542358, 36.173067, -5.462952, 36.969804)
aoi_crs = CRS.WGS84
resolution = 60
time_interval = ("2021-06-12", "2021-06-13")

aoi_bbox = BBox(bbox=aoi_coords_wgs84, crs=aoi_crs) # Bounding Box
aoi_size = bbox_to_dimensions(aoi_bbox, resolution=resolution)
print(f"Image shape at {resolution} m resolution: {aoi_size} pixels")

#%% CATALOG API

catalog = SentinelHubCatalog(config=config)

aoi_bbox = BBox(bbox=aoi_coords_wgs84, crs=aoi_crs) # Bounding Box
time_interval = time_interval

search_iterator = catalog.search(
    DataCollection.SENTINEL2_L2A,
    bbox=aoi_bbox,
    time=time_interval,
    fields={"include": ["id", "properties.datetime"], "exclude": []},

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
    evalscript=evalscript_true_color,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L2A.define_from(
                name="s2l2a", service_url="https://sh.dataspace.copernicus.eu"
            ),
            time_interval=time_interval,
            other_args={"dataFilter": {"mosaickingOrder": "leastCC"}},
        )
    ],
    responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
    bbox=aoi_bbox,
    size=aoi_size,
    config=config,
)

#%% Request
data_full_color = request_true_color.get_data()

#%% Plotting
plot_image(data_full_color[0], factor=3.5 / 255, clip_range=(0, 1))

# %%
