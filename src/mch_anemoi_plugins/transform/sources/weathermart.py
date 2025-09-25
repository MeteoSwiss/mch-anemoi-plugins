import datetime
from typing import Any
from itertools import chain

from anemoi.datasets.create.source import Source
from anemoi.datasets.dates.groups import GroupOfDates
import numpy as np
from pyproj import CRS
from weathermart import DataProvider
from weathermart.default_provider import available_retrievers, default_provider
from weathermart.utils import read_file
import xarray as xr

from mch_anemoi_plugins.xarray_extensions import CustomFieldList
from mch_anemoi_plugins.helpers import reproject


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

def get_all_available_sources() -> list[str]:
    return list(set(chain.from_iterable(r.sources for r in available_retrievers())))


def assign_lonlat(array: xr.DataArray, crs: str) -> xr.DataArray:
    if crs == "epsg:4326":
        # If the CRS is already WGS84, we can directly assign longitude and latitude
        return array.assign_coords(
            longitude=("x", array.x.data), latitude=("y", array.y.data)
        )
    xv, yv = np.meshgrid(array.x, array.y, indexing="ij")
    lon, lat = reproject(xv, yv, crs, CRS.from_user_input("epsg:4326"))
    return array.assign_coords(longitude=(("x", "y"), lon), latitude=(("x", "y"), lat))

def get_fieldlist_from_data_provider(
    provider: DataProvider,
    source: str,
    dates: list[datetime.datetime],
    param: list[str] | None = None,
    **retriever_kwargs: Any,
) -> CustomFieldList:
    expanded_kwargs = retriever_kwargs.copy()
    for k, v in retriever_kwargs.items():
        if isinstance(v, str) and v.startswith("$file:"):
            expanded_kwargs[k] = read_file(v.removeprefix("$file:"))
    data = provider.provide(source, param, dates, **expanded_kwargs)
    data = data.drop_duplicates(...)
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
    xarray_fieldlist = CustomFieldList.from_xarray(data, proj_string=crs, source=source)
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
    return {name: make_source_class(name) for name in source_names}


source_names = get_all_available_sources()
source_classes = get_all_source_classes(source_names)
# This will print the source class definitions
# Still have to execute this code to make sure the classes are created and available for the pyproject
print("\n".join(f'{name.replace("-", "_")} = make_source_class("{name}")' for name in source_names))

DHM25 = make_source_class("DHM25")
OPERA = make_source_class("OPERA")
NASADEM = make_source_class("NASADEM")
SATELLITE = make_source_class("SATELLITE")
SURFACE = make_source_class("SURFACE")