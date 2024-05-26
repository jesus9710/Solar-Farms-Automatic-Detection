from __future__ import annotations

from typing import Any

import os
from pathlib import Path
import tempfile

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt

from matplotlib.patches import Polygon as PltPolygon
from mpl_toolkits.basemap import Basemap  # Available here: https://github.com/matplotlib/basemap
from shapely.geometry import MultiLineString, MultiPolygon, Polygon, box, shape

from sentinelhub import SentinelHubRequest, CRS

import rioxarray
from rioxarray import merge

def plot_image(
    image: np.ndarray, factor: float = 1.0, clip_range: tuple[float, float] | None = None, **kwargs: Any
) -> None:
    """Utility function for plotting RGB images."""
    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
    if clip_range is not None:
        ax.imshow(np.clip(image * factor, *clip_range), **kwargs)
    else:
        ax.imshow(image * factor, **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])

def show_splitter(splitter, alpha=0.2, area_buffer=0.2, show_legend=False):
    area_bbox = splitter.get_area_bbox()
    minx, miny, maxx, maxy = area_bbox
    lng, lat = area_bbox.middle
    w, h = maxx - minx, maxy - miny
    minx = minx - area_buffer * w
    miny = miny - area_buffer * h
    maxx = maxx + area_buffer * w
    maxy = maxy + area_buffer * h

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    base_map = Basemap(
        projection="mill",
        lat_0=lat,
        lon_0=lng,
        llcrnrlon=minx,
        llcrnrlat=miny,
        urcrnrlon=maxx,
        urcrnrlat=maxy,
        resolution="l",
        epsg=4326,
    )
    base_map.drawcoastlines(color=(0, 0, 0, 0))

    area_shape = splitter.get_area_shape()

    if isinstance(area_shape, Polygon):
        polygon_iter = [area_shape]
    elif isinstance(area_shape, MultiPolygon):
        polygon_iter = area_shape.geoms
    else:
        raise ValueError(f"Geometry of type {type(area_shape)} is not supported")

    for polygon in polygon_iter:
        if isinstance(polygon.boundary, MultiLineString):
            for linestring in polygon.boundary:
                ax.add_patch(PltPolygon(np.array(linestring), closed=True, facecolor=(0, 0, 0, 0), edgecolor="red"))
        else:
            ax.add_patch(
                PltPolygon(np.array(polygon.boundary.coords), closed=True, facecolor=(0, 0, 0, 0), edgecolor="red")
            )

    bbox_list = splitter.get_bbox_list()
    info_list = splitter.get_info_list()

    cm = plt.get_cmap("jet", len(bbox_list))
    legend_shapes = []
    for i, bbox in enumerate(bbox_list):
        wgs84_bbox = bbox.transform(CRS.WGS84).get_polygon()

        tile_color = tuple(list(cm(i))[:3] + [alpha])
        ax.add_patch(PltPolygon(np.array(wgs84_bbox), closed=True, facecolor=tile_color, edgecolor="green"))

        if show_legend:
            legend_shapes.append(plt.Rectangle((0, 0), 1, 1, fc=cm(i)))

    if show_legend:
        legend_names = []
        for info in info_list:
            legend_name = "{},{}".format(info["index_x"], info["index_y"])

            for prop in ["grid_index", "tile"]:
                if prop in info:
                    legend_name = "{},{}".format(info[prop], legend_name)

            legend_names.append(legend_name)

        plt.legend(legend_shapes, legend_names)
    plt.tight_layout()
    plt.show()

def get_subarea(bbox, **kwargs):
    return SentinelHubRequest(bbox = bbox, data_folder=tempfile.gettempdir(), **kwargs)

def get_sh_request_from_split(bbox_splitted, **kwargs):
    bbox_list = bbox_splitted.get_bbox_list()
    sh_requests = [get_subarea(bbox, **kwargs) for bbox in bbox_list]
    dl_requests = [request.download_list[0] for request in sh_requests]

    for request in dl_requests:
        request.save_response = True

    return sh_requests, dl_requests

def get_product_path(sh_requests):
    data_folder = sh_requests[0].data_folder
    return [Path(data_folder) / req.get_filename_list()[0] for req in sh_requests]

def merge_products(products, file_name):

    rasters = []
    for prod in products:
        rasters.append(rioxarray.open_rasterio(prod))
    
    merged_raster = merge.merge_arrays(rasters)
    merged_raster.rio.to_raster(Path(os.getcwd().split(sep='src')[0]+'\\data\\'+file_name))