from copy import copy
from typing import Callable
import io

from meteodatalab import data_source, grib_decoder
import earthkit.data as ekd
import numpy as np
import xarray as xr
from pyproj import CRS
from pyproj import Transformer


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


class FieldListDataSource(data_source.DataSource):
    def __init__(self, fieldlist: ekd.FieldList):
        self.fieldlist = fieldlist

    def _retrieve(self, request: dict):
        yield from self.fieldlist.sel(**request)


def meteodatalab_wrapper(func: Callable[..., dict[str, xr.DataArray]]) -> Callable[[ekd.FieldList], ekd.FieldList]:
    """Decorator to wrap a function that processes an ekd.FieldList.
    """
    def inner(fieldlist: ekd.FieldList) -> ekd.FieldList:
        source = FieldListDataSource(fieldlist)
        result = func(source)
        return _meteodalab_ds_to_fieldlist(result)
    
    return inner

def to_meteodatalab(fieldlist: ekd.FieldList) -> dict[str, xr.DataArray]:
    """Convert an ekd.FieldList to a dictionary of xarray DataArrays."""
    source = FieldListDataSource(fieldlist)
    return grib_decoder.load(source, {})

def from_meteodatalab(ds: dict[str, xr.DataArray]) -> ekd.FieldList:
    """Convert a dictionary of xarray DataArrays to an ekd.FieldList."""
    return _meteodalab_ds_to_fieldlist(ds)

def _meteodalab_ds_to_fieldlist(ds: dict[str, xr.DataArray]) -> ekd.FieldList:
    with io.BytesIO() as buffer:
        
        # write data to the buffer
        for da in ds.values():
            grib_decoder.save(da, buffer, bits_per_value=32) # TODO: find out why we need 32 and 16 leads to precision loss
        
        # reset the buffer position to the beginning
        buffer.seek(0)
        
        # read data from the buffer into a FieldList
        fs = ekd.from_source("stream", buffer, read_all=True, lazily=False)
        
        # somehow read_all does not work correctly, so we need to convert to FieldList
        # to actually have all data loaded in memory and not get IO errors later
        fl = ekd.FieldList.from_fields(fs)

    return fl
