import datetime
import json
from itertools import chain
from typing import Any
from typing import Dict
from typing import List
from typing import Union

import numpy as np
import xarray as xr
import yaml
from anemoi.datasets.create.source import Source
from anemoi.datasets.create.sources.xarray_support.field import XArrayField
from anemoi.datasets.create.sources.xarray_support.fieldlist import XarrayFieldList
from anemoi.datasets.create.sources.xarray_support.flavour import CoordinateGuesser
from anemoi.datasets.create.sources.xarray_support.time import Time
from anemoi.datasets.create.sources.xarray_support.variable import Variable
from anemoi.datasets.dates.groups import GroupOfDates
from data_provider import DataProvider
from data_provider.default_provider import all_retrievers
from data_provider.default_provider import default_provider
from data_provider.utils import read_file
from gridefix_process.helpers import reproject
from pyproj import CRS


def get_all_data_provider_sources() -> List[str]:
    return list(set(chain.from_iterable(r.sources for r in all_retrievers())))


def assign_lonlat(array: xr.DataArray, crs: str) -> xr.DataArray:
    xv, yv = np.meshgrid(array.x, array.y, indexing="ij")
    lon, lat = reproject(xv, yv, crs, CRS.from_user_input("epsg:4326"))
    return array.assign_coords(longitude=(("x", "y"), lon), latitude=(("x", "y"), lat))


class MCHVariable(Variable):
    """A variable class for MCH data, extending the XArrayVariable class with more metadata."""

    def __init__(
        self,
        *,
        ds: xr.Dataset,
        var: xr.DataArray,
        coordinates: List[Any],
        grid: xr.Dataset,
        time: Time,
        metadata: Dict[Any, Any],
        proj_string: Union[str, None] = None,
        source: str = "",
        **kwargs: Any,
    ) -> None:
        """
        Initialize MCHVariable.

        Args:
            ds (xr.Dataset): Input dataset.
            var (xr.DataArray): Variable data array.
            coordinates (List[Any]): List of coordinates.
            grid (xr.Dataset): Grid dataset.
            time (Time): Time object.
            metadata (Dict[Any, Any]): Metadata dictionary.
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

    def __getitem__(self, i: int) -> "MCHField":
        if i >= self.length:
            raise IndexError(i)
        coords = np.unravel_index(i, self.shape)
        kwargs = {k: v for k, v in zip(self.names, coords)}
        return MCHField(self, self.variable.isel(kwargs))


class MCHField(XArrayField):
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
    def not_grid_dim(self) -> List[str]:
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
        """
        Retrieve the coordinate reference system (CRS) string.

        This is derived from the projection string.

        """
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


class MCHFieldList(XarrayFieldList):
    @classmethod
    def from_xarray(
        cls,
        ds: xr.Dataset,
        flavour: Union[str, dict, None] = None,
        proj_string: Union[str, None] = None,
        source: str = "",
    ) -> "MCHFieldList":
        """
        Create an MCHFieldList from an xarray dataset.

        Args:
            ds (xr.Dataset): Input xarray dataset.
            flavour (Union[str, dict, None], optional): Flavour configuration. Default is None.
            proj_string (Union[str, None], optional): Projection string. Default is None.
            source (str, optional): Source string. Default is "".

        Returns:
            MCHFieldList: An instance of MCHFieldList populated with variables from the dataset.
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
                MCHVariable(
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
    """ Helpers function to check and remove unsupported coordinates and dimensions. 
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


def get_fieldlist_from_data_provider(
    provider: DataProvider,
    source: str,
    dates: List[datetime.datetime],
    param: Union[List[str], None] = None,
    **retriever_kwargs: Any,
) -> MCHFieldList:
    expanded_kwargs = retriever_kwargs.copy()
    for k, v in retriever_kwargs.items():
        if isinstance(v, str) and v.startswith("$file:"):
            expanded_kwargs[k] = read_file(v.removeprefix("$file:"))
    data = provider.provide(source, param, dates, **expanded_kwargs)
    time_dim = (
        "forecast_reference_time"
        if "forecast_reference_time" in data.dims
        else "time"
        if "time" in data.dims
        else None
    )
    crs = provider.get_crs(source)
    if isinstance(crs, dict) and "through" in expanded_kwargs:
        crs = crs[expanded_kwargs["through"]]
    elif isinstance(crs, dict) and "gridefix" in crs:
        crs = crs["gridefix"]  # retrievers default for NWP and satellite data
    if time_dim is not None:  # e.g. not a forcing dataset
        data = data.sortby(time_dim).sel(
            {time_dim: dates}, method="nearest"
        )  # return only the relevant time for daily arrays
    else:
        data = data.assign_coords(
            time=dates
        )  # if no time dimension, assign the dates as a coordinate
        time_dim = "time"
    if not ("longitude" in data.coords and "latitude" in data.coords):
        data = assign_lonlat(data, crs)
    if (
        len(dates) == 1
    ):  # minimal input selects first day at midnight even if time is missing...
        first_hour_day = [d for d in data[time_dim].dt.round("1d").to_numpy()]
        data = data.reindex_like(
            xr.Dataset(coords={time_dim: first_hour_day}), method="nearest"
        )
    data = check_indexing(data, time_dim)
    xarray_fieldlist = MCHFieldList.from_xarray(data, proj_string=crs, source=source)
    return xarray_fieldlist


class DataProviderSource(Source):
    """Base source class for data provider sources in Anemoi."""

    def __init__(self, context, source, param, **retriever_kwargs):
        super().__init__(
            context, source, param=param, retriever_kwargs=retriever_kwargs
        )
        self.provider = default_provider()
        self.source = source
        self.param = param
        self.retriever_kwargs = retriever_kwargs

    def execute(self, dates: list[datetime.datetime]):
        if isinstance(dates, GroupOfDates):
            dates = dates.dates
        if not isinstance(self.param, list):
            self.param = [self.param]
        fieldlist = get_fieldlist_from_data_provider(
            self.provider,
            self.source,
            dates,
            self.param,
            **self.retriever_kwargs,
        )
        return fieldlist


def make_source_class(source_name: str):
    def __init__(self, context, param, **retriever_kwargs):
        super(self.__class__, self).__init__(
            context, source_name, param, **retriever_kwargs
        )

    class_name = source_name.replace("-", "_")
    return type(class_name, (DataProviderSource,), {"__init__": __init__})


def get_all_source_classes(source_names: list[str]) -> dict[str, type]:
    """
    Create source classes for all specified source names.

    Args:
        source_names (list[str]): A list of source names to create classes for.

    Returns:
        dict[str, Type[DataProviderSource]]: A dictionary mapping source names to their
            dynamically created classes.
    """
    return {name: make_source_class(name) for name in source_names}


source_names = get_all_data_provider_sources()
source_classes = get_all_source_classes(source_names)
# This will print the source class definitions
# Still have to execute this code to make sure the classes are created and available for the pyproject
print(
    "\n".join(
        f'{name.replace("-", "_")} = make_source_class("{name}")'
        for name in source_names
    )
)
COSMO_1E = make_source_class("COSMO-1E")
INCA = make_source_class("INCA")
OPERA = make_source_class("OPERA")
GEOSATCLIM = make_source_class("GEOSATCLIM")
KENDA_CH1 = make_source_class("KENDA-CH1")
RADAR = make_source_class("RADAR")
DHM25 = make_source_class("DHM25")
CEDTM = make_source_class("CEDTM")
CMSAF = make_source_class("CMSAF")
SATELLITE = make_source_class("SATELLITE")
NASADEM = make_source_class("NASADEM")
ICON_CH1_EPS = make_source_class("ICON-CH1-EPS")
SURFACE = make_source_class("SURFACE")
