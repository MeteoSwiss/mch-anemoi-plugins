import datetime
import json
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from anemoi.datasets.create.sources.xarray_support.field import XArrayField
from anemoi.datasets.create.sources.xarray_support.fieldlist import XarrayFieldList
from anemoi.datasets.create.sources.xarray_support.flavour import CoordinateGuesser
from anemoi.datasets.create.sources.xarray_support.time import Time
from anemoi.datasets.create.sources.xarray_support.variable import Variable
from pyproj import CRS

from mch_anemoi_plugins.helpers import assign_lonlat, reproject


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
    """
    A variable container for MCH datasets.

    MCHVariable wraps an xarray DataArray along with metadata, coordinate information,
    grid information, and temporal context. It also stores an optional projection string
    and source identifier for the variable. This class is used to facilitate data processing
    for MCH datasets by providing methods to index into the variable data and to apply
    consistent metadata transformations.

    Attributes:
        proj_string (Union[str, None]): Projection string associated with the variable.
        source (str): Identifier of the data source.
    """

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
        self.proj_string: Union[str, None] = proj_string
        self.source: str = source
        self._metadata = {
            x.replace("variable", "param"): k for x, k in self._metadata.items()
        }

    def __getitem__(self, i: int) -> "MCHField":
        """
        Get item by index.

        Args:
            i (int): Index.

        Returns:
            MCHField: MCHField object corresponding to the index.
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
        Get the source string for this field.
        """
        return self.owner.source

    @property
    def proj_string(self) -> str:
        """
        Retrieve the projection string set in the owner.
        """
        return self.owner.proj_string

    @property
    def grid_coords(self) -> np.ndarray:
        """
        Determine grid coordinates from the data array's dimensions and coordinates.

        It finds an intersection between standard grid coordinates (x, y, longitude, latitude)
        and typical station/cell identifiers.
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
        Identify dimensions that are not part of the grid.
        """
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


def provide_to_fieldset(source: str) -> Any:
    """
    Provide data to a fieldset.

    Args:
        source (str): Source string.

    Returns:
        function: An entrypoint function that accepts context, dates, and retriever parameters,
                  returning an MCHFieldList.
    """

    def anemoi_entrypoint(
        context: Any,
        dates: List[datetime.datetime],
        param: Union[List[str], None] = None,
        **retriever_kwargs: Any,
    ) -> MCHFieldList:
        """
        Anemoi entrypoint function.

        Args:
            context (Any): Context object.
            dates (List[datetime.datetime]): List of dates.
            param (Union[List[str], None], optional): List of parameters. Default is None.
            **retriever_kwargs (Any): Additional retriever keyword arguments.

        Returns:
            MCHFieldList: Field list created from the provided data.
        """
        from data_provider.default_provider import default_provider
        from data_provider.utils import read_file

        provider = default_provider()

        expanded_kwargs = retriever_kwargs.copy()
        for k, v in retriever_kwargs.items():
            if isinstance(v, str) and v.startswith("$file:"):
                expanded_kwargs[k] = read_file(v.removeprefix("$file:"))
        data = provider.provide(source, param, dates, **expanded_kwargs)

        # Get the CRS from the provider and adjust to the source ('through') provided.
        crs = provider.get_crs(source)
        if isinstance(crs, dict) and "through" in expanded_kwargs:
            crs = crs[expanded_kwargs["through"]]

        # Determine the time dimension
        if "time" in data.dims:
            time_dim = "time"
        elif "forecast_reference_time" in data.dims:
            time_dim = "forecast_reference_time"
        else:
            time_dim = None

        # Select data based on the time dimension
        # or if forcing assign a time coordinate.
        if time_dim is not None:
            data = data.sel({time_dim: dates}, method="nearest")
        else:
            data = data.assign_coords(forecast_reference_time=dates)
            time_dim = "forecast_reference_time"

        # Ensure latitude and longitude coordinates are available.
        if not ("longitude" in data.coords and "latitude" in data.coords):
            data = assign_lonlat(data, crs)

        # If only one date is provided, reindex using the relevant dateime (default by anemoi-dataset is to use the midnight value, not always available).
        if len(dates) == 1:
            first_hour_day = [d for d in data[time_dim].dt.round("1d").to_numpy()]
            data = data.reindex_like(
                xr.Dataset(coords={time_dim: first_hour_day}), method="nearest"
            )

        # Drop coordinates that may mislead the dataset guesser if they are not dimensions.
        for coord in [
            "forecast_reference_time",
            "realization",
            "step",
            "surface_altitude",
            "land_area_fraction",
        ]:
            if coord in data.coords and coord not in data.dims:
                data = data.drop(coord)

        # Update the 'lead_time' attribute if the dimension exists.
        if "lead_time" in data.dims:
            data["lead_time"].attrs.update(standard_name="forecast_period")

        # Ensure that the 'number' dimension exists.
        if "number" not in data.dims:
            data = data.expand_dims(number=[0])

        # Reindex the 'forecast_reference_time' coordinate with ISO-formatted dates.
        isodates = [pd.to_datetime(d).isoformat() for d in dates]
        time_ds = xr.Dataset(coords={"forecast_reference_time": isodates})
        data = data.rename({time_dim: "forecast_reference_time"})
        data = data.reindex_like(time_ds, method="bfill")
        data["forecast_reference_time"].attrs.update(standard_name="time")

        # If spatial dimensions are present, transpose the data into a standard order.
        if "x" in data.dims:
            data = data.transpose("forecast_reference_time", "number", "x", "y")
        xarray_fieldlist = MCHFieldList.from_xarray(
            data, proj_string=crs, source=source
        )
        return xarray_fieldlist

    return anemoi_entrypoint


radar = provide_to_fieldset("RADAR")
inca = provide_to_fieldset("INCA")
station = provide_to_fieldset("SURFACE")
dem = provide_to_fieldset("NASADEM")
satellite = provide_to_fieldset("SATELLITE")
geosatclim = provide_to_fieldset("GEOSATCLIM")
opera = provide_to_fieldset("OPERA")
