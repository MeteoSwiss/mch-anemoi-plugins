import datetime
import json
from typing import List, Union

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from anemoi.datasets.create.sources.xarray_support.field import XArrayField
from anemoi.datasets.create.sources.xarray_support.fieldlist import XarrayFieldList
from anemoi.datasets.create.sources.xarray_support.flavour import CoordinateGuesser
from anemoi.datasets.create.sources.xarray_support.time import Time
from anemoi.datasets.create.sources.xarray_support.variable import Variable
from data_provider.default_provider import default_provider
from data_provider.utils import read_file
from mch_anemoi_plugins.helpers import reproject, assign_lonlat
from pyproj import CRS


def align_dates_with_freq(
    data: xr.DataArray, dates: List[datetime.datetime], freq: str = "5min"
) -> xr.DataArray:
    """
    Align dates in the data array with a specified frequency.

    Args:
        data (xr.DataArray): Input data array.
        dates (List[datetime.datetime]): List of dates to align.
        freq (str): Frequency to align to. Default is "5min".

    Returns:
        xr.DataArray: Data array with aligned dates.
    """
    ideal = pd.date_range(
        pd.to_datetime(dates[0]), pd.to_datetime(dates[-1]), freq=freq
    )
    data["forecast_reference_time"] = ideal
    return data


class MCHVariable(Variable):
    def __init__(
        self,
        *,
        ds: xr.Dataset,
        var: xr.DataArray,
        coordinates: List,
        grid: xr.Dataset,
        time: Time,
        metadata: dict,
        proj_string: str = None,
        source: str = "",
        **kwargs,
    ):
        """
        Initialize MCHVariable.

        Args:
            ds (xr.Dataset): Dataset.
            var (xr.DataArray): Variable data array.
            coordinates (List): List of coordinates.
            grid (xr.Dataset): Grid dataset.
            time (Time): Time object.
            metadata (dict): Metadata dictionary.
            proj_string (str, optional): Projection string. Default is None.
            source (str, optional): Source string. Default is "".
            **kwargs: Additional keyword arguments.
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
        """
        Get item by index.

        Args:
            i (int): Index.

        Returns:
            MCHField: MCHField object.
        """
        if i >= self.length:
            raise IndexError(i)
        coords = np.unravel_index(i, self.shape)
        kwargs = {k: v for k, v in zip(self.names, coords)}
        return MCHField(self, self.variable.isel(kwargs))


class MCHField(XArrayField):
    @property
    def source(self) -> str:
        """
        Get source.

        Returns:
            str: Source string.
        """
        return self.owner.source

    @property
    def proj_string(self) -> str:
        """
        Get projection string.

        Returns:
            str: Projection string.
        """
        return self.owner.proj_string

    @property
    def grid_coords(self) -> np.ndarray:
        """
        Get grid coordinates.

        Returns:
            np.ndarray: Array of grid coordinates.
        """
        std_grid_coords = ["x", "y", "longitude", "latitude"]
        station_or_cell = ["cell", "station"]
        data_coords = [c for c in self.selection.coords]
        data_dims = [c for c in self.selection.dims]
        return np.intersect1d(
            std_grid_coords + station_or_cell, data_coords + data_dims
        )

    @property
    def not_grid_dim(self) -> List[str]:
        """
        Get non-grid dimensions.

        Returns:
            List[str]: List of non-grid dimensions.
        """
        return [d for d in self.selection.dims if d not in self.grid_coords]

    @property
    def resolution(self) -> str:
        """
        Get resolution.

        Returns:
            str: Resolution string.
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
        Get coordinate reference system.

        Returns:
            str: CRS string.
        """
        return self.proj_string

    @property
    def bounding_box(self) -> tuple:
        """
        Get bounding box.

        Returns:
            tuple: Bounding box coordinates.
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
        flavour: Union[str, dict] = None,
        proj_string: str = None,
        source: str = "",
    ) -> "MCHFieldList":
        """
        Create MCHFieldList from xarray dataset.

        Args:
            ds (xr.Dataset): Input dataset.
            flavour (Union[str, dict], optional): Flavour configuration. Default is None.
            proj_string (str, optional): Projection string. Default is None.
            source (str, optional): Source string. Default is "".

        Returns:
            MCHFieldList: MCHFieldList object.
        """
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


def provide_to_fieldset(source: str):
    """
    Provide data to fieldset.

    Args:
        source (str): Source string.

    Returns:
        function: Anemoi entrypoint function.
    """
    provider = default_provider()

    def anemoi_entrypoint(
        context,
        dates: List[datetime.datetime],
        param: Union[List[str], None] = None,
        **retriever_kwargs,
    ) -> MCHFieldList:
        """
        Anemoi entrypoint function.

        Args:
            context: Context object.
            dates (List[datetime.datetime]): List of dates.
            param (Union[List[str], None], optional): List of parameters. Default is None.
            **retriever_kwargs: Additional retriever keyword arguments.

        Returns:
            MCHFieldList: MCHFieldList object.
        """
        expanded_kwargs = retriever_kwargs.copy()
        for k, v in retriever_kwargs.items():
            if isinstance(v, str) and v.startswith("$file:"):
                expanded_kwargs[k] = read_file(v.removeprefix("$file:"))
        data = provider.provide(source, param, dates, **expanded_kwargs)
        crs = provider.get_crs(source)
        time_dim = (
            "time"
            if "time" in data.dims
            else "forecast_reference_time"
            if "forecast_reference_time" in data.dims
            else None
        )
        if time_dim is None:
            data = data.assign_coords(forecast_reference_time=dates)
        elif time_dim == "time":
            data = data.rename({time_dim: "forecast_reference_time"})
        time_dim = "forecast_reference_time"
        data = data.sel({time_dim: dates}, method="nearest")
        if crs is None:
            crs = "epsg:2056"
        if not ("longitude" in data.coords and "latitude" in data.coords):
            data = assign_lonlat(data, crs)
        if len(dates) == 1:
            first_hour_day = [d for d in data[time_dim].dt.round("1d").to_numpy()]
            data = data.reindex_like(
                xr.Dataset(coords={time_dim: first_hour_day}), method="nearest"
            )
        potentially_misleading_coords = [
            "forecast_reference_time",
            "realization",
            "step",
            "surface_altitude",
            "land_area_fraction",
        ]
        for n in potentially_misleading_coords:
            if n in data.coords and n not in data.dims:
                data = data.drop(n)
        if "lead_time" in data.dims:
            data["lead_time"].attrs.update(standard_name="forecast_period")
        if "number" not in data.dims:
            data = data.expand_dims(number=[0])
        isodates = [pd.to_datetime(d).isoformat() for d in dates]
        x = xr.Dataset(coords=dict(forecast_reference_time=isodates))
        data = data.reindex_like(x, method="bfill")
        data["forecast_reference_time"].attrs.update(standard_name="time")
        if "x" in data.dims:
            data = data.transpose("forecast_reference_time", "number", "x", "y")
        xarray_fieldlist = MCHFieldList.from_xarray(
            data, proj_string=crs, source=source
        )
        return xarray_fieldlist

    return anemoi_entrypoint


cosmo = provide_to_fieldset("COSMO-1E")
radar = provide_to_fieldset("RADAR")
icon = provide_to_fieldset("ICON-CH1-EPS")
inca = provide_to_fieldset("INCA")
station = provide_to_fieldset("SURFACE")
dem = provide_to_fieldset("NASADEM")
satellite = provide_to_fieldset("SATELLITE")
kenda = provide_to_fieldset("KENDA-CH1")
geosatclim = provide_to_fieldset("GEOSATCLIM")
opera = provide_to_fieldset("OPERA")
