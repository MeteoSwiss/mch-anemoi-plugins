from typing import Any

from anemoi.datasets.create.sources.xarray_support.field import XArrayField
from anemoi.datasets.create.sources.xarray_support.fieldlist import XarrayFieldList
from anemoi.datasets.create.sources.xarray_support.flavour import CoordinateGuesser
from anemoi.datasets.create.sources.xarray_support.time import Time
from anemoi.datasets.create.sources.xarray_support.variable import Variable
import numpy as np
import xarray as xr
import yaml
import json
from pyproj import CRS

from mch_anemoi_plugins.helpers import reproject

class CustomVariable(Variable):
    """A variable class for MCH data, extending the XArrayVariable class with more metadata."""

    def __init__(
        self,
        *,
        ds: xr.Dataset,
        var: xr.DataArray,
        coordinates: list[Any],
        grid: xr.Dataset,
        time: Time,
        metadata: dict[Any, Any],
        proj_string: str | None = None,
        source: str = "",
        **kwargs: Any,
    ) -> None:
        """
        Initialize CustomVariable.

        Args:
            ds (xr.Dataset): Input dataset.
            var (xr.DataArray): Variable data array.
            coordinates (list[Any]): List of coordinates.
            grid (xr.Dataset): Grid dataset.
            time (Time): Time object.
            metadata (dict[Any, Any]): Metadata dictionary.
            proj_string (Union[str, None], optional): Projection string. Default is None.
            source (str, optional): Source string. Default is "".
            **kwargs (Any): Additional keyword arguments.

        Returns:
            None
        """
        super().__init__(
            ds=ds,
            variable=var,
            coordinates=coordinates,
            grid=grid,
            time=time,
            metadata=metadata,
            **kwargs,
        )
        self.proj_string = proj_string
        self.source = source
        self._metadata = {
            x.replace("variable", "param"): k for x, k in self._metadata.items()
        }

    def __getitem__(self, i: int) -> "CustomField":
        if i >= self.length:
            raise IndexError(i)
        coords = np.unravel_index(i, self.shape)
        kwargs = {k: v for k, v in zip(self.names, coords)}
        return CustomField(self, self.variable.isel(kwargs))


class CustomField(XArrayField):
    @property
    def source(self) -> str:
        return self.owner.source

    @property
    def proj_string(self) -> str:
        return self.owner.proj_string

    @property
    def grid_coords(self) -> np.ndarray:
        std_grid_coords = ["x", "y", "longitude", "latitude"]
        station_or_cell = ["cell", "station"]
        data_coords = [c for c in self.selection.coords]
        data_dims = [c for c in self.selection.dims]
        return np.intersect1d(
            std_grid_coords + station_or_cell, data_coords + data_dims
        )

    @property
    def not_grid_dim(self) -> list[str]:
        return [d for d in self.selection.dims if d not in self.grid_coords]

    @property
    def resolution(self) -> str:
        """
        Compute the resolution based on the minimal spacing along grid dimensions.

        For projected CRS, it computes the minimal difference in x and y (converted to meters)
        and rounds it to a kilometer value.
        """
        if "crs" in self.selection.attrs:
            valid_crs = CRS.from_user_input(self.selection.attrs["crs"])
        else:
            valid_crs = CRS.from_user_input(self.crs)
        valid_crs = CRS.from_user_input("epsg:2056")
        minimal = self.selection.isel({d: 0 for d in self.not_grid_dim}).rio.write_crs(
            valid_crs
        )
        if minimal.rio.crs.is_projected:
            minx = minimal.x.to_numpy()
            minx.sort()
            minx = np.diff(minx).min()
            miny = minimal.y.to_numpy()
            miny.sort()
            miny = np.diff(miny).min()
            res_m = np.array([minx, miny]) * minimal.rio.crs.units_factor[1]
        else:
            res_deg = np.array(minimal.rio.resolution())
            res_m = (
                res_deg
                * np.diff(
                    reproject(
                        [0, 1], [0, 0], valid_crs, CRS.from_user_input("epsg:2056")
                    )
                )[0][0]
            )
        res_km = np.round(res_m / 1e3, 0)
        return f"{tuple(v.item() for v in res_km)[0]}km"

    @property
    def crs(self) -> str:
        return self.proj_string

    @property
    def bounding_box(self) -> tuple:
        """
        Compute the bounding box of the field as (min_x, min_y, max_x, max_y).

        It selects the minimal values along non-grid dimensions and calculates the extent.
        """
        minimal = self.selection.isel({d: 0 for d in self.not_grid_dim})
        bbox = (
            float(np.min(minimal.x)),
            float(np.min(minimal.y)),
            float(np.max(minimal.x)),
            float(np.max(minimal.y)),
        )
        return bbox


class CustomFieldList(XarrayFieldList):
    @classmethod
    def from_xarray(
        cls,
        ds: xr.Dataset,
        flavour: str | dict | None = None,
        proj_string: str | None = None,
        source: str = "",
    ) -> "CustomFieldList":
        """
        Create an CustomFieldList from an xarray dataset.

        Returns:
            CustomFieldList: An instance of CustomFieldList populated with variables from the dataset.
        """
        variables = []
        if isinstance(flavour, str):
            with open(flavour) as f:
                if flavour.endswith((".yaml", ".yml")):
                    flavour = yaml.safe_load(f)
                else:
                    flavour = json.load(f)
        guess = CoordinateGuesser.from_flavour(ds, flavour)
        skip = set()

        def _skip_attr(v: Any, attr_name: str) -> None:
            attr_val = getattr(v, attr_name, "")
            if isinstance(attr_val, str):
                skip.update(attr_val.split(" "))

        for name in ds.data_vars:
            v = ds[name]
            _skip_attr(v, "coordinates")
            _skip_attr(v, "bounds")
            _skip_attr(v, "grid_mapping")
        # Select only geographical variables
        for name in ds.data_vars:
            if name in skip:
                continue
            v = ds[name]
            v.attrs.update(crs=proj_string)
            coordinates = []
            for coord in v.coords:
                c = guess.guess(ds[coord], coord)
                assert c, f"Could not guess coordinate for {coord}"
                if coord not in v.dims:
                    c.is_dim = False
                coordinates.append(c)
            variables.append(
                CustomVariable(
                    ds=ds,
                    var=v,
                    coordinates=coordinates,
                    grid=guess.grid(coordinates, variable=v),
                    time=Time.from_coordinates(coordinates),
                    metadata={},
                    proj_string=proj_string,
                    source=source,
                )
            )
        return cls(ds, variables)


def check_indexing(data: xr.Dataset, time_dim: str) -> xr.Dataset:
    """Helpers function to check and remove unsupported coordinates and dimensions.
    In particular, in the case of analysis data from gridefix, 1 lead time of 0 is provided as a coord, while the forecast_reference_time is the time dimension. However, anemoi does not support lead_time as a coordinate if only one value is provided (the ndarray of lead times has dim 0). We remove lead_time and set the time dimension to forecast_reference_time.
    Otherwise, observation data is indexed by time, and forecast data by forecast_reference_time and lead_time, which anemoi supports."""
    potentially_misleading_coords = [
        "surface_altitude",
        "land_area_fraction",
        "stationName",
        "dataOwner",
    ]  # those will make anemoi-datasets display warnings for unsupported coordinates
    for n in potentially_misleading_coords:
        if n in data.coords and n not in data.dims:
            data = data.drop(n)
    if "number" not in data.dims:
        if "realization" in data.dims:
            # realization dimension is used for ensemble members
            # but anemoi expects a number dimension
            data = data.rename({"realization": "number"})
        else:
            data = data.expand_dims(number=[0])
    if time_dim == "forecast_reference_time":
        if "lead_time" in data.dims:
            # forecast data
            data["forecast_reference_time"].attrs.update(standard_name="date")
            data["lead_time"].attrs.update(standard_name="forecast_period")
        else:
            # wrongly indexed observation/analysis data
            if "lead_time" in data.coords:
                data = data.drop_vars("lead_time")  # misleading for anemoi, will try to
                # interpret it as forecast data
            if "time" in data.coords:
                data = data.drop_vars(
                    "time"
                )  # remove time coordinate otherwise 2 coordinates are intepreted as time and anemoi complains
            data["forecast_reference_time"].attrs.update(standard_name="time")
    elif time_dim == "time":
        # observation data
        data["time"].attrs.update(standard_name="time")
    return data