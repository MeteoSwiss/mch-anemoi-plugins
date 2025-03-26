from copy import copy

import numpy as np
from pyproj import CRS, Transformer


def reproject(x_coords, y_coords, src_crs, dst_crs):
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return transformer.transform(x_coords, y_coords)


def replace(instance, **kwargs):
    new_instance = copy(instance)
    for k, v in kwargs.items():
        if hasattr(new_instance, k):
            setattr(new_instance, k, v)
        else:
            raise AttributeError(f"Attribute '{k}' does not exist in the instance.")
    return new_instance


def assign_lonlat(array, crs):
    xv, yv = np.meshgrid(array.x, array.y, indexing="ij")
    lon, lat = reproject(xv, yv, crs, CRS.from_user_input("epsg:4326"))
    return array.assign_coords(longitude=(("x", "y"), lon), latitude=(("x", "y"), lat))
