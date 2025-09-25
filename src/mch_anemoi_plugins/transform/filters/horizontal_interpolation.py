import re
from typing import Union

import dask.array as da
import numpy as np
import xarray as xr
from anemoi.transform.filter import Filter
from earthkit.data.indexing.fieldlist import FieldArray  
from pyproj import CRS
from scipy.interpolate import NearestNDInterpolator
from scipy.spatial import cKDTree

from mch_anemoi_plugins.helpers import assign_lonlat
from mch_anemoi_plugins.helpers import reproject
from mch_anemoi_plugins.xarray_extensions import CustomFieldList


def merge_fieldlist(field_array: FieldArray) -> xr.Dataset:
    """Merge a FieldArray into a single xarray.Dataset grouped by forecast_reference_time."""
    times = sorted(
        set(field.metadata("forecast_reference_time") for field in field_array)
    )
    merged = []
    for t in times:
        datasets = [
            field.selection.to_dataset()
            for field in field_array
            if field.metadata("forecast_reference_time") == t
        ]
        merged.append(xr.merge(datasets))
    return xr.concat(merged, dim="forecast_reference_time")


def _interp2grid(
    array: xr.Dataset, example_field, template: Union[xr.Dataset, str], method="linear"
) -> xr.Dataset:
    from gridefix_process import grid_interp # type: ignore

    if isinstance(template, str) and template.startswith("$file:"):
        template = xr.open_zarr(template.removeprefix("$file:"))
    if "cell" in array.dims:
        array_reduced = interp2points(
            template, array, source_crs=example_field.crs, target_crs=template.crs
        )
        array = array_reduced.set_xindex(["x", "y"]).unstack()
    ds_from_array = array.assign_attrs(
        {"source": example_field.source, "crs": example_field.crs}
    )
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


def _interp_na(array: xr.Dataset, param: str) -> xr.Dataset:
    da_vals = array[param].to_numpy()
    indices = np.where(np.isfinite(da_vals))
    nans = np.isnan(da_vals)
    interp = NearestNDInterpolator(np.transpose(indices), da_vals[indices])
    da_vals[nans] = interp(*np.where(nans))
    array[param].data = da_vals
    return array


def _interp2res(
    array: xr.Dataset, example_field, resolution: Union[str, int], target_crs=None
) -> xr.Dataset:
    from gridefix_process import grid_interp # type: ignore

    resolution_km = float(re.sub(r"[^0-9.\-]", "", str(resolution)))
    target_crs = target_crs or example_field.crs
    _xmin, _ymin, _xmax, _ymax = example_field.bounding_box
    if target_crs != example_field.crs:
        [_xmin, _xmax], [_ymin, _ymax] = reproject(
            [_xmin, _xmax],
            [_ymin, _ymax],
            CRS.from_user_input(example_field.crs),
            CRS.from_user_input(target_crs),
        )
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
            x=(
                np.arange(_xmin, _xmax, resolution_in_crs_units, dtype=np.float64)
                + resolution_in_crs_units / 2
            ),
            y=(
                np.arange(_ymin, _ymax, resolution_in_crs_units, dtype=np.float64)
                + resolution_in_crs_units / 2
            ),
        ),
        attrs={"crs": target_crs},
    )
    if "cell" in array.dims:
        array_reduced = interp2points(template, array, example_field.crs, target_crs)
        array = array_reduced.set_xindex(["x", "y"]).unstack()
    ds_from_array = array.assign_attrs(
        {"source": example_field.source, "crs": target_crs}
    )
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


def _project(array: xr.Dataset, example_field, target_crs) -> xr.Dataset:
    src_crs = CRS.from_user_input(example_field.crs)
    dest_crs = CRS.from_user_input(target_crs)
    if "station" in array.dims or "cell" in array.dims:
        new_x, new_y = reproject(array.x, array.y, src_crs, dest_crs)
        new_array = array.reset_coords(["x", "y"], drop=True).assign_coords(
            x=(array.dims, new_x), y=(array.dims, new_y)
        )
    else:
        xv, yv = np.meshgrid(array.x, array.y, indexing="ij")
        new_x, new_y = reproject(xv, yv, src_crs, dest_crs)
        new_array = array.reset_index(array.dims).assign_coords(
            x=new_x[:, 0], y=new_y[0]
        )
    new_array.attrs["crs"] = target_crs
    return new_array


def interp2points(
    template: xr.Dataset, array: xr.Dataset, source_crs, target_crs
) -> xr.Dataset:
    array = array.sortby(["x", "y"])
    meshx, meshy = np.meshgrid(template.x.data, template.y.data)
    flat_x, flat_y = meshx.flatten(), meshy.flatten()
    template_coords = da.stack([flat_x, flat_y], axis=1)
    projected_x, projected_y = reproject(
        array.x.data, array.y.data, source_crs, target_crs
    )
    data_coords = da.stack([projected_x, projected_y], axis=1)
    tree = cKDTree(data_coords)
    _, indices = tree.query(template_coords)
    array_reduced = array.isel(cell=indices)
    array_reduced = array_reduced.assign_coords(x=("cell", flat_x), y=("cell", flat_y))
    return array_reduced



class BaseXarrayFilter(Filter):
    api_version = "1.0.0"
    schema = None

    def forward(self, field_array: FieldArray) -> FieldArray:
        """Merge fields, apply the filter, and return a new FieldArray."""
        example = field_array[0]
        merged = merge_fieldlist(field_array)
        result = self.apply_filter(merged, example)
        return CustomFieldList.from_xarray(
            result, proj_string=example.crs, source=example.source
        )

    def apply_filter(self, ds: xr.Dataset, example_field) -> xr.Dataset:
        """Override in subclass with the actual transformation."""
        raise NotImplementedError



class Interp2Grid(BaseXarrayFilter):
    """Interpolate fields to a target grid."""

    def __init__(self, template: Union[xr.Dataset, str], method: str = "linear"):
        self.template = template
        self.method = method

    def apply_filter(self, ds: xr.Dataset, example_field) -> xr.Dataset:
        return _interp2grid(
            ds, example_field, template=self.template, method=self.method
        )


class InterpNAFilter(BaseXarrayFilter):
    """Fill NaN values for a given parameter using nearest neighbor interpolation."""

    def __init__(self, param: str):
        self.param = param

    def apply_filter(self, ds: xr.Dataset, example_field) -> xr.Dataset:
        return _interp_na(ds, self.param)


class Interp2Res(BaseXarrayFilter):
    """Interpolate fields to a target resolution."""

    def __init__(self, resolution: Union[str, int], target_crs: str = None):
        self.resolution = resolution
        self.target_crs = target_crs

    def apply_filter(self, ds: xr.Dataset, example_field) -> xr.Dataset:
        return _interp2res(
            ds, example_field, resolution=self.resolution, target_crs=self.target_crs
        )


class Project(BaseXarrayFilter):
    """Project fields to a target coordinate reference system."""

    def __init__(self, target_crs: str):
        self.target_crs = target_crs

    def apply_filter(self, ds: xr.Dataset, example_field) -> xr.Dataset:
        return _project(ds, example_field, target_crs=self.target_crs)