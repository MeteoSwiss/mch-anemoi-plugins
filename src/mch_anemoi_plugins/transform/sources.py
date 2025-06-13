import datetime
import json
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from anemoi.datasets.create.source import Source
from anemoi.datasets.create.sources.xarray_support.field import XArrayField
from anemoi.datasets.create.sources.xarray_support.fieldlist import XarrayFieldList
from anemoi.datasets.create.sources.xarray_support.flavour import CoordinateGuesser
from anemoi.datasets.create.sources.xarray_support.time import Time
from anemoi.datasets.create.sources.xarray_support.variable import Variable
from data_provider.default_provider import default_provider
from data_provider.utils import read_file
from gridefix_process.helpers import reproject
from pyproj import CRS


def get_opera_target_grid():
    # target grid: Regular lat lon grid
    custom_target_grid = xr.Dataset(
        {"x": np.arange(0.5, 16.5, 0.01), "y": np.arange(43, 50, 0.01)},
        attrs={
            "grid_mapping": {
                "epsg_code": "4326",
            },
            "source": "Custom grid",
        },
    )
    return custom_target_grid


def assign_lonlat(array, crs):
    xv, yv = np.meshgrid(array.x, array.y, indexing="ij")
    lon, lat = reproject(xv, yv, crs, CRS.from_user_input("epsg:4326"))
    return array.assign_coords(longitude=(("x", "y"), lon), latitude=(("x", "y"), lat))


class MCHVariable(Variable):

    def __init__(
        self,
        *,
        ds,
        var,
        coordinates,
        grid,
        time,
        metadata,
        proj_string=None,
        source="",
        **kwargs,
    ):
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
        self._metadata = {x.replace("variable", "param"): k for x, k in self._metadata.items()}

    def __getitem__(self, i):
        if i >= self.length:
            raise IndexError(i)
        coords = np.unravel_index(i, self.shape)
        kwargs = {k: v for k, v in zip(self.names, coords)}
        return MCHField(self, self.variable.isel(kwargs))


class MCHField(XArrayField):
    @property
    def source(self):
        return self.owner.source

    @property
    def proj_string(self):
        return self.owner.proj_string

    @property
    def grid_coords(self):
        std_grid_coords = ["x", "y", "longitude", "latitude"]
        station_or_cell = ["cell", "station"]
        data_coords = [c for c in self.selection.coords]
        data_dims = [c for c in self.selection.dims]
        return np.intersect1d(std_grid_coords + station_or_cell, data_coords + data_dims)

    @property
    def not_grid_dim(self):
        return [d for d in self.selection.dims if d not in self.grid_coords]

    @property
    def resolution(self):
        if "crs" in self.selection.attrs:
            valid_crs = CRS.from_user_input(self.selection.attrs["crs"])
        else:
            valid_crs = CRS.from_user_input(self.crs)
        valid_crs = CRS.from_user_input("epsg:2056")
        minimal = self.selection.isel({d: 0 for d in self.not_grid_dim}).rio.write_crs(valid_crs)
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
            res_m = res_deg * np.diff(reproject([0, 1], [0, 0], valid_crs, CRS.from_user_input("epsg:2056")))[0][0]
        res_km = np.round(res_m / 1e3, 0)
        return f"{tuple(v.item() for v in res_km)[0]}km"

    @property
    def crs(self):
        return self.proj_string

    @property
    def bounding_box(self):
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
    def from_xarray(cls, ds, flavour=None, proj_string=None, source=""):
        variables = []
        if isinstance(flavour, str):
            with open(flavour) as f:
                if flavour.endswith(".yaml") or flavour.endswith(".yml"):
                    flavour = yaml.safe_load(f)
                else:
                    flavour = json.load(f)
        guess = CoordinateGuesser.from_flavour(ds, flavour)
        skip = set()

        def _skip_attr(v, attr_name):
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


def get_fieldlist_from_data_provider(provider, source, dates, param, **retriever_kwargs):
    expanded_kwargs = retriever_kwargs.copy()
    for k, v in retriever_kwargs.items():
        if isinstance(v, str) and v.startswith("$file:"):
            expanded_kwargs[k] = read_file(v.removeprefix("$file:"))
    data = provider.provide(source, param, dates, **expanded_kwargs)
    time_dim = (
        "forecast_reference_time" if "forecast_reference_time" in data.dims else "time" if "time" in data.dims else None
    )
    data = data.drop_duplicates(...)
    crs = provider.get_crs(source)
    if isinstance(crs, dict) and "through" in expanded_kwargs:
        crs = crs[expanded_kwargs["through"]]
    if time_dim is not None:  # e.g. not a forcing dataset
        data = data.sortby(time_dim).sel(
            {time_dim: dates}, method="nearest"
        )  # return only the relevant time for daily arrays
    else:
        data = data.assign_coords(time=dates)
        crs = "epsg:2056"
    if not ("longitude" in data.coords and "latitude" in data.coords):
        data = assign_lonlat(data, crs)
    if len(dates) == 1:  # minimal input selects first day at midnight even if time is missing...
        first_hour_day = [d for d in data[time_dim].dt.round("1d").to_numpy()]
        data = data.reindex_like(xr.Dataset(coords={time_dim: first_hour_day}), method="nearest")
    potentially_misleading_coords = [
        "lead_time",
        "forecast_reference_time",
        "realization",
        "step",
        "surface_altitude",
        "land_area_fraction",
        "stationName",
    ]  # those are handled in a separate logic, we should probably change that and let anemoi deal with it
    for n in potentially_misleading_coords:
        if n in data.coords and n not in data.dims:
            data = data.drop(n)
    data = data.rename({time_dim: "forecast_reference_time"})
    if data.forecast_reference_time.to_series().duplicated().any() and source != "SATELLITE":
        data = data.groupby("forecast_reference_time").mean()  # avoids duplicated dates...
    if "lead_time" in data.dims:
        # forecast data
        data["lead_time"].attrs.update(standard_name="forecast_period")
    if "number" not in data.dims:
        data = data.expand_dims(number=[0])
    isodates = [pd.to_datetime(d).isoformat() for d in dates]
    x = xr.Dataset(coords=dict(forecast_reference_time=isodates))
    data = data.reindex_like(x, method="bfill")
    data["forecast_reference_time"].attrs.update(standard_name="time")
    if "x" in data.dims:
        data = data.transpose("forecast_reference_time", "number", "x", "y")
    xarray_fieldlist = MCHFieldList.from_xarray(data, proj_string=crs, source=source)
    return xarray_fieldlist


class DataProviderSource(Source):
    def __init__(self, context, source, param, **retriever_kwargs):
        super().__init__(context, source, param=param, retriever_kwargs=retriever_kwargs)
        self.provider = default_provider()
        self.source = source
        self.param = param
        self.retriever_kwargs = retriever_kwargs

    def execute(self, dates: list[datetime.datetime]):
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


class COSMO1E(DataProviderSource):
    def __init__(self, context, param, **retriever_kwargs):
        super().__init__(context, "COSMO-1E", param, **retriever_kwargs)


class RADAR(DataProviderSource):
    def __init__(self, context, param, **retriever_kwargs):
        super().__init__(context, "RADAR", param, **retriever_kwargs)


class ICON_CH1_EPS(DataProviderSource):
    def __init__(self, context, param, **retriever_kwargs):
        super().__init__(context, "ICON-CH1-EPS", param, **retriever_kwargs)


class INCA(DataProviderSource):
    def __init__(self, context, param, **retriever_kwargs):
        super().__init__(context, "INCA", param, **retriever_kwargs)


class SURFACE(DataProviderSource):
    def __init__(self, context, param, **retriever_kwargs):
        super().__init__(context, "SURFACE", param, **retriever_kwargs)


class NASADEM(DataProviderSource):
    def __init__(self, context, param, **retriever_kwargs):
        super().__init__(context, "NASADEM", param, **retriever_kwargs)


class SATELLITE(DataProviderSource):
    def __init__(self, context, param, **retriever_kwargs):
        super().__init__(context, "SATELLITE", param, **retriever_kwargs)


class KENDA_CH1(DataProviderSource):
    def __init__(self, context, param, **retriever_kwargs):
        super().__init__(context, "KENDA-CH1", param, **retriever_kwargs)


class GEOSATCLIM(DataProviderSource):
    def __init__(self, context, param, **retriever_kwargs):
        super().__init__(context, "GEOSATCLIM", param, **retriever_kwargs)


class OPERA(DataProviderSource):
    def __init__(self, context, param, **retriever_kwargs):
        super().__init__(context, "OPERA", param, **retriever_kwargs)
