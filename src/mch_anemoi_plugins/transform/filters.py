"""Plugins for anemoi-transform's filters."""

import re
from copy import copy
from typing import Union

import dask
import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from earthkit.data.core.fieldlist import Field
from earthkit.data.indexing.fieldlist import FieldArray
from earthkit.meteo import thermo
from earthkit.meteo.wind.array import polar_to_xy, xy_to_polar
from gridefix_process import grid_interp
from gridefix_process.helpers import reproject
from gridefix_process.timeseries import interptogranularity
from pyproj import CRS
from scipy.interpolate import NearestNDInterpolator
from scipy.spatial import cKDTree

from mch_anemoi_plugins.sources import MCHFieldList

dask.config.set({"array.chunk-size": "256MiB"})


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


def interp2points(template, array, source_crs, target_crs):
    array = array.sortby(["x", "y"])
    meshx, meshy = np.meshgrid(template.x.data, template.y.data)
    flat_x, flat_y = meshx.flatten(), meshy.flatten()
    template_coords = da.stack([flat_x, flat_y], axis=1)
    projected_x, projected_y = reproject(array.x.data, array.y.data, source_crs, target_crs)
    data_coords = da.stack([projected_x, projected_y], axis=1)
    tree = cKDTree(data_coords)
    # Query the tree to find the nearest neighbors for the template coordinates
    distances, indices = tree.query(template_coords)
    # Use the found indices to select the corresponding cells
    array_reduced = array.isel(cell=indices)
    array_reduced = array_reduced.assign_coords(x=("cell", flat_x), y=("cell", flat_y))
    return array_reduced


def _noop(field: Field):
    return field


def _q2dewpoint(array, example_field, q="QV_2M", sp="SP", dewpoint="TD_2M"):
    q_sl = array[q].data
    sp_sl = array[sp].data
    td_sl = thermo.dewpoint_from_specific_humidity(q=q_sl, p=sp_sl)
    new_array = array.assign({dewpoint: (array.dims, td_sl)}).drop_vars([q, sp])
    return new_array


def _dirspeed2uv(
    array,
    example_field,
    wind_speed="SP_10M",
    wind_dir="DD_10M",
    u_component="U_10M",
    v_component="V_10M",
    in_radians=False,
):
    wind_dir_data = array[wind_dir].data
    if in_radians:
        wind_dir_data = np.rad2deg(wind_dir_data)
    u, v = polar_to_xy(array[wind_speed].data, wind_dir_data)
    new_array = array.assign({u_component: (array.dims, u), v_component: (array.dims, v)}).drop_vars(
        [wind_speed, wind_dir]
    )
    return new_array


def _uv2dirspeed(
    array,
    example_field,
    u_component="U_10M",
    v_component="V_10M",
    wind_speed="SP_10M",
    wind_dir="DD_10M",
    in_radians=False,
):
    magnitude, direction = xy_to_polar(array[u_component].data, array[v_component].data)
    if in_radians:
        direction = np.deg2rad(direction)
    new_array = array.assign({wind_speed: (array.dims, magnitude), wind_dir: (array.dims, direction)}).drop_vars(
        [u_component, v_component]
    )
    return new_array


def _interp2grid(array, example_field, template: Union[xr.Dataset, str], method="linear"):
    if isinstance(template, str) and template.startswith("$file:"):
        template = xr.open_zarr(template.removeprefix("$file:"))
    kenda_data = "cell" in array.dims
    if kenda_data:
        # too many points, so we have to reduce before reindexing using gridefix
        array_reduced = interp2points(template, array, source_crs=example_field.crs, target_crs=template.crs)
        array = array_reduced.set_xindex(["x", "y"]).unstack()
    ds_from_array = array.assign_attrs({"source": example_field.source}).assign_attrs({"crs": example_field.crs})
    interpolated_array = (
        grid_interp.interp2grid(ds_from_array, dst_grid=template, method=method)
        .sortby("y", ascending=True)
        .chunk("auto")
    )
    interpolated_array = interpolated_array.assign_attrs({"crs": template.crs})
    for v in interpolated_array.data_vars:
        interpolated_array[v].attrs["crs"] = template.crs
    interpolated_array = assign_lonlat(interpolated_array, template.crs)
    if method != "nearest":
        interpolated_array = interpolated_array.interpolate_na("x", method=method)
    return interpolated_array.transpose("forecast_reference_time", "number", "x", "y")


def fillna(array, param):
    da = array[param].to_numpy()
    indices = np.where(np.isfinite(da))
    nans = np.isnan(da)
    interp = NearestNDInterpolator(np.transpose(indices), da[indices])
    da[nans] = interp(*np.where(nans))
    array[param].data = da
    return array


def _interp2res(array, example_field, resolution: Union[str, int], target_crs=None):
    resolution_km = float(re.sub("[^0-9.\-]", "", resolution))
    target_crs = target_crs or example_field.crs
    _xmin, _ymin, _xmax, _ymax = example_field.bounding_box
    if not target_crs == example_field.crs:
        [_xmin, _xmax], [_ymin, _ymax] = reproject(
            [_xmin, _xmax],
            [_ymin, _ymax],
            CRS.from_user_input(example_field.crs),
            CRS.from_user_input(target_crs),
        )
    # resolution is in km, we need to convert it to the units of the target crs
    resolution_in_crs_units = np.diff(
        reproject(
            [0, 1000],
            [0, 0],
            CRS.from_user_input("epsg:2056"),
            CRS.from_user_input(target_crs),
        )
    )[0][0]
    template = xr.Dataset(
        coords=dict(
            x=(np.arange(_xmin, _xmax, resolution_in_crs_units, dtype=np.float64) + resolution_in_crs_units / 2),
            y=(np.arange(_ymin, _ymax, resolution_in_crs_units, dtype=np.float64) + resolution_in_crs_units / 2),
        ),
        attrs={"crs": target_crs},
    )
    kenda_data = "cell" in array.dims
    if kenda_data:
        # too many points, so we have to reduce before reindexing using gridefix
        array_reduced = interp2points(template, array, example_field.crs, target_crs)
        array = array_reduced.set_xindex(["x", "y"]).unstack()
    ds_from_array = array.assign_attrs({"source": example_field.source}).assign_attrs({"crs": target_crs})
    interpolated_array = (
        grid_interp.interp2grid(ds_from_array, dst_grid=template, method="linear")
        .sortby("y", ascending=True)
        .chunk("auto")
    )
    interpolated_array = assign_lonlat(interpolated_array, target_crs)
    interpolated_array = interpolated_array.interpolate_na("x")
    interpolated_array.attrs["resolution"] = resolution_km
    interpolated_array.attrs["crs"] = target_crs
    return interpolated_array


def _project(array, example_field, target_crs):
    crs = example_field.crs
    src_CRS = CRS.from_user_input(crs)
    dest_CRS = CRS.from_user_input(target_crs)
    xindexes = array.dims
    station_data = "station" in xindexes or "cell" in xindexes
    if station_data:
        new_x, new_y = reproject(array.x, array.y, src_CRS, dest_CRS)
        new_array = array.reset_coords(["x", "y"], drop=True).assign_coords(x=(xindexes, new_x), y=(xindexes, new_y))
    else:
        xv, yv = np.meshgrid(array.x, array.y, indexing="ij")
        new_x, new_y = reproject(xv, yv, src_CRS, dest_CRS)
        # basically interpolate to nearest point
        new_x_list = new_x[:, 0]
        new_y_list = new_y[0]
        new_array = array.reset_index(xindexes).assign_coords(x=new_x_list, y=new_y_list)
    new_array.attrs["crs"] = target_crs
    return new_array


def xarray_filter(op):

    def anemoi_op(ctx, field_array: FieldArray, *args, **kwargs):
        print(f"Applying {op.__name__.replace('_', '')} filter")
        example_field = field_array[0]
        concat = []
        for d in np.unique([field.metadata("forecast_reference_time") for field in field_array]):
            concat.append(
                xr.merge(
                    [
                        field.selection.to_dataset()
                        for field in field_array
                        if field.metadata("forecast_reference_time") == d
                    ]
                )
            )
        all_fields = xr.concat(concat, dim="forecast_reference_time")
        kwargs.update(example_field=example_field)
        res = op(array=all_fields, *args, **kwargs)
        target_crs = kwargs.get("target_crs", example_field.crs)
        print(f"Finished applying {op.__name__.replace('_', '')} filter")
        fieldlist = MCHFieldList.from_xarray(res, proj_string=target_crs, source=example_field.source)
        return fieldlist

    return anemoi_op


def anemoi_filter(op):

    def anemoi_op(ctx, field_array: FieldArray, *args, **kwargs):
        print(f"Applying {op.__name__.replace('_', '')} filter")
        transformed_fields = []
        for field in field_array:
            field: Field
            res = op(field, *args, **kwargs)
            if res is not None:
                transformed_fields.append(res)
        return FieldArray(transformed_fields)

    return anemoi_op


interp2res = xarray_filter(_interp2res)
project = xarray_filter(_project)
interp2grid = xarray_filter(_interp2grid)
noop = anemoi_filter(_noop)
uv2dirspeed = xarray_filter(_uv2dirspeed)
dirspeed2uv = xarray_filter(_dirspeed2uv)
