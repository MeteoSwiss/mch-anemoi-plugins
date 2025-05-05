from copy import copy

import earthkit.data as ekd
import numpy as np
from anemoi.transform.fields import new_field_from_numpy, new_fieldlist_from_list
from anemoi.transform.filter import Filter


def reproject(x_coords, y_coords, src_crs, dst_crs):
    from pyproj import Transformer

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
    from pyproj import CRS

    xv, yv = np.meshgrid(array.x, array.y, indexing="ij")
    lon, lat = reproject(xv, yv, crs, CRS.from_user_input("epsg:4326"))
    return array.assign_coords(longitude=(("x", "y"), lon), latitude=(("x", "y"), lat))
