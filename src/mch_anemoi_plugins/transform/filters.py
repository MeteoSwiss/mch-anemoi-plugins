"""
Plugin filters for anemoi-transform.
Each filter is implemented as a plugin class that provides a
forward(data: FieldList) -> FieldList method.
"""

import re
from typing import Union

import dask.array as da
import earthkit.data as ekd
import numpy as np
import xarray as xr
from anemoi.transform.fields import new_field_from_numpy
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.transform.filter import Filter
from earthkit.data.core import Field  # adjust as needed
from earthkit.data.indexing.fieldlist import FieldArray  # adjust as needed
from earthkit.meteo import thermo
from earthkit.meteo.wind.array import polar_to_xy
from earthkit.meteo.wind.array import xy_to_polar
from pyproj import CRS
from scipy.interpolate import NearestNDInterpolator
from scipy.spatial import cKDTree

from mch_anemoi_plugins.helpers import assign_lonlat
from mch_anemoi_plugins.helpers import reproject
from mch_anemoi_plugins.transform.sources import MCHFieldList


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


def _q2dewpoint(array: xr.Dataset, q="QV_2M", sp="SP", dewpoint="TD_2M") -> xr.Dataset:
    q_sl = array[q].data
    sp_sl = array[sp].data
    td_sl = thermo.dewpoint_from_specific_humidity(q=q_sl, p=sp_sl)
    new_array = array.assign({dewpoint: (array.dims, td_sl)}).drop_vars([q, sp])
    return new_array


def _dirspeed2uv(
    array: xr.Dataset,
    wind_speed="SP_10M",
    wind_dir="DD_10M",
    u_component="U_10M",
    v_component="V_10M",
    in_radians=False,
) -> xr.Dataset:
    wind_dir_data = array[wind_dir].data
    if in_radians:
        wind_dir_data = np.rad2deg(wind_dir_data)
    u, v = polar_to_xy(array[wind_speed].data, wind_dir_data)
    new_array = array.assign(
        {u_component: (array.dims, u), v_component: (array.dims, v)}
    ).drop_vars([wind_speed, wind_dir])
    return new_array


def _uv2dirspeed(
    array: xr.Dataset,
    u_component="U_10M",
    v_component="V_10M",
    wind_speed="SP_10M",
    wind_dir="DD_10M",
    in_radians=False,
) -> xr.Dataset:
    magnitude, direction = xy_to_polar(array[u_component].data, array[v_component].data)
    if in_radians:
        direction = np.deg2rad(direction)
    new_array = array.assign(
        {wind_speed: (array.dims, magnitude), wind_dir: (array.dims, direction)}
    ).drop_vars([u_component, v_component])
    return new_array


def _interp2grid(
    array: xr.Dataset, example_field, template: Union[xr.Dataset, str], method="linear"
) -> xr.Dataset:
    from gridefix_process import grid_interp

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
    from gridefix_process import grid_interp

    resolution_km = float(re.sub("[^0-9.\-]", "", str(resolution)))
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


def _destagger_field(field: Field, dim: str) -> xr.DataArray:
    import meteodatalab.operators.destagger as dsg
    from meteodatalab.grib_decoder import _FieldBuffer
    from meteodatalab.grib_decoder import _is_ensemble

    buffer = _FieldBuffer(_is_ensemble(field))
    buffer.load(field, None)
    da = buffer.to_xarray()

    # TODO: this hardcoding is not ideal
    if f"origin_{dim}" not in da.attrs:
        da.attrs[f"origin_{dim}"] = 0.5

    return dsg.destagger(da, dim).squeeze(drop=True)


class HorizontalDestagger(Filter):
    """A filter to destagger fields using meteodata-lab."""

    # @matching(select="param", forward=("param",))
    def __init__(self, param_dim: dict[str, str]):
        """Initialize the filter.

        Parameters
        ----------
        param_dim:
            Dictionary mapping parameter names to dimensions along which to destagger.
        """
        self.param_dim = param_dim
        self.param = list(param_dim.keys())

    # def forward_transform(self, *fields: ekd.Field) -> tp.Iterator[Field]:
    #     """Destagger the field."""

    #     for field in fields:
    #         param = field.metadata("param")
    #         yield self.new_field_from_numpy(
    #             _destagger_mdl(field, self.param_dim[param]).values,
    #             template=field,
    #             param=param
    #         )

    def forward(self, data: ekd.FieldList) -> ekd.FieldList:
        result = []
        for field in data:
            if (param := field.metadata("param")) in self.param_dim:
                field = new_field_from_numpy(
                    _destagger_field(field, self.param_dim[param]).values,
                    template=field,
                    param=param,
                )
            result.append(field)
        return new_fieldlist_from_list(result)

    def backward_transform(self):
        raise NotImplementedError("HorizontalDestagger is not reversible.")


# --- Base plugin class for xarray-based filters ---
class BaseXarrayFilter(Filter):
    api_version = "1.0.0"
    schema = None

    def forward(self, field_array: FieldArray) -> FieldArray:
        """Merge fields, apply the filter, and return a new FieldArray."""
        example = field_array[0]
        merged = merge_fieldlist(field_array)
        result = self.apply_filter(merged, example)
        return MCHFieldList.from_xarray(
            result, proj_string=example.crs, source=example.source
        )

    def apply_filter(self, ds: xr.Dataset, example_field) -> xr.Dataset:
        """Override in subclass with the actual transformation."""
        raise NotImplementedError


# --- Plugin filter classes ---


class Q2Dewpoint(BaseXarrayFilter):
    """Compute dewpoint from specific humidity and pressure."""

    def __init__(self, q: str = "QV_2M", sp: str = "SP", dewpoint: str = "TD_2M"):
        self.q = q
        self.sp = sp
        self.dewpoint = dewpoint

    def apply_filter(self, ds: xr.Dataset, example_field) -> xr.Dataset:
        return _q2dewpoint(ds, q=self.q, sp=self.sp, dewpoint=self.dewpoint)


class Dirspeed2UV(BaseXarrayFilter):
    """Convert wind speed/direction to U/V components."""

    def __init__(
        self,
        wind_speed: str = "SP_10M",
        wind_dir: str = "DD_10M",
        u_component: str = "U_10M",
        v_component: str = "V_10M",
        in_radians: bool = False,
    ):
        self.wind_speed = wind_speed
        self.wind_dir = wind_dir
        self.u_component = u_component
        self.v_component = v_component
        self.in_radians = in_radians

    def apply_filter(self, ds: xr.Dataset, example_field) -> xr.Dataset:
        return _dirspeed2uv(
            ds,
            wind_speed=self.wind_speed,
            wind_dir=self.wind_dir,
            u_component=self.u_component,
            v_component=self.v_component,
            in_radians=self.in_radians,
        )


class UV2Dirspeed(BaseXarrayFilter):
    """Convert U/V components to wind speed/direction."""

    def __init__(
        self,
        u_component: str = "U_10M",
        v_component: str = "V_10M",
        wind_speed: str = "SP_10M",
        wind_dir: str = "DD_10M",
        in_radians: bool = False,
    ):
        self.u_component = u_component
        self.v_component = v_component
        self.wind_speed = wind_speed
        self.wind_dir = wind_dir
        self.in_radians = in_radians

    def apply_filter(self, ds: xr.Dataset, example_field) -> xr.Dataset:
        return _uv2dirspeed(
            ds,
            u_component=self.u_component,
            v_component=self.v_component,
            wind_speed=self.wind_speed,
            wind_dir=self.wind_dir,
            in_radians=self.in_radians,
        )


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
