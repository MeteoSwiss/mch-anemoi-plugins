# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import earthkit.data as ekd
import xarray as xr
from anemoi.transform.filter import Filter

from meteodatalab.operators import vertical_interpolation, vertical_extrapolation

from mch_anemoi_plugins.helpers import to_meteodatalab, from_meteodatalab
from datetime import datetime, timedelta
from typing import Any
import logging
import abc
from concurrent.futures import ThreadPoolExecutor

import earthkit.data as ekd
import numpy as np
from anemoi.transform.fields import new_field_from_grid
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.transform.flavour import RuleBasedFlavour
from anemoi.transform.grids import grid_registry

from anemoi.datasets.create.typing import DateList

from anemoi.datasets.create.source import Source
from anemoi.datasets.create.sources import source_registry

from mch_anemoi_plugins.utils.grib import decode_step, encode_step

VARIABLES_PROPERTIES = {
    "ALB_RAD": {"paramId": 500056, "type": "inst"},
    "ASWDIFD_S": {"paramId": 500481, "type": "accum-from-start", "window": "T"},
    "ASWDIFU_S": {"paramId": 500482, "type": "accum-from-start", "window": "T"},
    "ASWDIFU_S_OS": {"paramId": 503756, "type": "accum-from-start", "window": "T"},
    "ASWDIR_S": {"paramId": 500480, "type": "accum-from-start", "window": "T"},
    "ASWDIR_S_OS": {"paramId": 503752, "type": "accum-from-start", "window": "T"},
    "CAPE_ML": {"paramId": 500153, "type": "inst"},
    "CAPE_MU": {"paramId": 500151, "type": "inst"},
    "CIN_ML": {"paramId": 500154, "type": "inst"},
    "CLCH": {"paramId": 500050, "type": "inst"},
    "CLCL": {"paramId": 500048, "type": "inst"},
    "CLCM": {"paramId": 500049, "type": "inst"},
    "CLCT": {"paramId": 500046, "type": "inst"},
    "DBZ_CMAX": {"paramId": 500175, "type": "inst"},
    "DHAIL_MX": {"paramId": 503557, "type": "aggr-max", "window": "10m"},
    "DURSUN": {"paramId": 500584, "type": "accum-from-start", "window": "T"},
    "EDP": {"paramId": 503675, "type": "inst"},
    "FI": {"paramId": 500006, "type": "inst"},
    "H_SNOW": {"paramId": 500045, "type": "inst"},
    "HHL": {"paramId": 500008, "type": "inst", "vertStag": True},
    "HPBL": {"paramId": 502757, "type": "inst"},
    "HZEROCL": {"paramId": 500127, "type": "inst"},
    "LFC_ML": {"paramId": 503187, "type": "inst"},
    "LPI": {"paramId": 503142, "type": "inst"},
    "LPI_MAX": {"paramId": 503348, "type": "aggr-max", "window": "10m"},
    "P": {"paramId": 500001, "type": "inst"},
    "PMSL": {"paramId": 500002, "type": "inst"},
    "PS": {"paramId": 500000, "type": "inst"},
    "QC": {"paramId": 500100, "type": "inst"},
    "QV": {"paramId": 500035, "type": "inst"},
    "RELHUM": {"paramId": 500037, "type": "inst"},
    "RUNOFF_G": {"paramId": 500066, "type": "accum-from-start", "window": "T"},
    "RUNOFF_S": {"paramId": 500068, "type": "accum-from-start", "window": "T"},
    "SLI": {"paramId": 503204, "type": "inst"},
    "SNOWLMT": {"paramId": 500128, "type": "inst"},
    "T": {"paramId": 500014, "type": "inst"},
    "T_2M": {"paramId": 500011, "type": "inst"},
    "T_G": {"paramId": 500010, "type": "inst"},
    "T_SO": {"paramId": 500166, "type": "inst"},
    "THETAE": {"paramId": 500303, "type": "inst"},
    "TD_2M": {"paramId": 500017, "type": "inst"},
    "TOT_PR": {"paramId": 503303, "type": "inst"},
    "TOT_PREC": {"paramId": 500041, "type": "accum-from-start", "window": "T"},
    "TQC": {"paramId": 500051, "type": "inst"},
    "TQG": {"paramId": 500107, "type": "inst"},
    "TQI": {"paramId": 500040, "type": "inst"},
    "TQR": {"paramId": 500104, "type": "inst"},
    "TQS": {"paramId": 500105, "type": "inst"},
    "TWATER": {"paramId": 500108, "type": "inst"},
    "U": {"paramId": 500028, "type": "inst"},
    "U_10M": {"paramId": 500027, "type": "inst"},
    "U_10M_AV": {"paramId": 503743, "type": "aggr-avg", "window": "10m"},
    "V": {"paramId": 500030, "type": "inst"},
    "V_10M": {"paramId": 500029, "type": "inst"},
    "V_10M_AV": {"paramId": 503744, "type": "aggr-avg", "window": "10m"},
    "VMAX_10M": {"paramId": 500164, "type": "aggr-max", "window": "10m"},
    "W": {"paramId": 500032, "type": "inst", "vertStag": True},
    "W_SO": {"paramId": 500167, "type": "inst"},
    "WSHEAR_DIFF": {"paramId": 503228, "type": "inst"},
}

BASE_FDB_REQUEST = {
    "stream": "reanl",
    "class": "rd",
    "expver": "r001",
    "model": "icon-rea-l-ch1",
    "type": "cf",
}


ICON_GRID_PATH = "/scratch/mch/jenkins/icon/pool/data/ICON/mch/grids/icon-1/icon_grid_0001_R19B08_mch.nc"


LOG = logging.getLogger(__name__)

def realch1_source_router(context: Any, request: dict[str, Any], **kwargs: dict[str, Any]) -> Source:
    """Route to the appropriate REA-L-CH1 data source type based on the requested variables."""

    # determine source type and check consistency
    source_type = set(VARIABLES_PROPERTIES[var]["type"] for var in request["param"])
    if len(source_type) > 1:
        msg = "Expected all variables to have the same source type, got: \n"
        for st in source_type:
            vars_of_type = [var for var in request["param"] if VARIABLES_PROPERTIES[var]["type"] == st]
            msg += f" - {st}: {vars_of_type}\n"
        raise ValueError(msg)
    source_type = source_type.pop()

    # route to the appropriate source class
    match source_type:
        case "inst":
            return _Instantaneous(context, request, **kwargs)
        case "accum-from-start":
            return _AccumulationFromStart(context, request, **kwargs)
        case "aggr-avg":
            return _AggregationAverage(context, request, **kwargs)
        case _:
            raise ValueError(f"Unsupported variable type: {VARIABLES_PROPERTIES[request['param']]['type']}")


    
class ReaLCh1Base(Source):
    """REA-L-CH1 data source base."""

    emoji = "💽"

    def __init__(
        self,
        context,
        request: dict[str],
        **kwargs: dict[str, Any],
    ):
        """Initialise the FDB input.

        Parameters
        ----------
        context : dict
            The context.
        param : dict[str, str]
            The parameter request dictionary.
        kwargs : dict[str, Any]
            Additional keyword arguments that are specific for the requested parameter type.
        """
        super().__init__(context)

        self._processing_request = None

        self.request = BASE_FDB_REQUEST | request.copy()
        self.request["param"] = [VARIABLES_PROPERTIES[var]["paramId"] for var in request["param"]]

        # requests for pressure levels are deferred to processing step (interpolation)
        # so the initial request is modified to get model levels instead
        if request["levtype"] == "pl":
            self.request["levtype"] = "ml"
            self.request["levelist"] = "1/to/81"
            self._processing_request = {"levtype": request["levtype"], "levelist": request["levelist"]}
        
        self.grid = grid_registry.from_config({"icon": {"path": ICON_GRID_PATH}})
        # self.mixins_kwargs = kwargs

    def assign_grid(self, fl: ekd.FieldList) -> ekd.FieldList:
        """Assign the ICON grid to the fields in the FieldList.

        Parameters
        ----------
        fl : ekd.FieldList
            The input FieldList.

        Returns
        -------
        ekd.FieldList
            The FieldList with the grid assigned.
        """
        return new_fieldlist_from_list([new_field_from_grid(field, self.grid) for field in fl])
    
    @abc.abstractmethod
    def prepare_request(self, datetimes: list[datetime]) -> dict[str, Any]:
        pass
    
    def process_data(self, fl: ekd.FieldList) -> ekd.FieldList:

        out_fl = ekd.SimpleFieldList()
        for paramgroup in fl.group_by("param"):
            date, time, step = set(paramgroup.metadata("date", "time", "step")).pop()
            if self._processing_request and self._processing_request["levtype"] == "pl":
                pressure_fields = ekd.from_source("fdb", self.request | {
                        "param": 500001,
                        "date": date,
                        "time": time,
                        "step": step,
                    }
                )
                pressure_fields = next(iter(to_meteodatalab(pressure_fields).values()))
                paramgroup = to_meteodatalab(paramgroup)
                paramgroup = _interpolate_to_pressure_levels(
                    paramgroup,
                    pressure_fields,
                    [float(lvl) for lvl in self._processing_request["levelist"]],
                )
                paramgroup = from_meteodatalab(paramgroup)
            for field in paramgroup.order_by("valid_time"):
                field = override_time_metadata(field)
                out_fl.append(field)
        return out_fl


    def execute(self, dates: list[datetime]) -> ekd.FieldList:
        request = self.prepare_request(dates)
        self._last_request = request
        LOG.info("Submitting FDB request: %s", request)
        now = datetime.now()
        fl = ekd.from_source("fdb", request)
        LOG.info("FDB request completed in %s seconds.", (datetime.now() - now).total_seconds())
        fl = self.process_data(fl)
        fl = self.assign_grid(fl)
        return fl


class _Instantaneous(ReaLCh1Base):
    """Instantaneous variable source."""

    @staticmethod
    def time_request(datetimes: list[datetime]) -> dict[str, Any]:
        """Defines the time-related keys for the FDB request."""
        dates = set()
        steps = set()        
        for dt in datetimes:
            dates.add(dt.strftime("%Y%m%d"))
            steps.add(int((dt - dt.replace(hour=0, minute=0)).total_seconds() // 3600))
        return {
            "date": sorted(dates),
            "time": "0000",
            "step": sorted(steps),
        }
    
    def prepare_request(self, dates: list[datetime]) -> dict[str, Any]: 
        return self.request | self.time_request(dates)

class _AccumulationFromStart(ReaLCh1Base):

    def __init__(self, context, request: dict[str], accumulation_period_hours: int = 1):
        self.accumulation_period_hours = accumulation_period_hours
        super().__init__(context, request)
        
    @staticmethod
    def time_request(datetimes: list[datetime]) -> dict[str, Any]:
        """Defines the time-related keys for the FDB request."""

        # the accumulated precipitation for 00UTC of a given day, is actually
        # the accumulation between the "24UTC" (+24h step) and the 23UTC (+23h step)
        # of the previous day
        dates = set()
        steps = set()
        for dt in datetimes:
            if dt.hour == 0 and dt.minute == 0:
                dates.add((dt - timedelta(days=1)).strftime("%Y%m%d"))
                steps.add(24)
                continue
            dates.add(dt.strftime("%Y%m%d"))
            steps.add(int((dt - dt.replace(hour=0, minute=0)).total_seconds() // 3600))
        
        return {
            "date": sorted(dates),
            "time": "0000",
            "step": sorted(steps),
        }
    
    def prepare_request(self, dates: list[datetime]) -> dict[str, Any]:
        """Ensure unique and sorted dates."""

        if len(set(dates)) != len((dates := sorted(dates))):
            raise ValueError("dates must be unique and sorted.")
        
        # add previous datetimes for accumulation
        accum_steps = []
        for step in range(1, self.accumulation_period_hours + 1):
            accum_steps.append(dates[0] - timedelta(hours=step))
        dates = sorted(set(dates) | set(accum_steps))

        # prepare request
        time_request = self.time_request(dates)
        return self.request | time_request

    def process_data(self, fl: ekd.FieldList) -> ekd.FieldList:
        accum_fl = ekd.SimpleFieldList()
        for paramgroup in fl.group_by("param"):
            for field in paramgroup.order_by("valid_time")[::-1]:
                field = override_time_metadata(field)
                valid_time = datetime.strptime(field.metadata("valid_time"), "%Y-%m-%dT%H:%M:%S")
                accum_start_valid_time = valid_time - timedelta(hours=self.accumulation_period_hours)
                accum_start_field = paramgroup.sel(valid_time=datetime.strftime(
                    accum_start_valid_time,
                    "%Y-%m-%dT%H:%M:%S",
                ))
                if len(accum_start_field) == 0:
                    # for now, skip fields where we don't have the previous accumulation start field
                    # a check is done later to ensure all expected fields are present
                    continue  
                if len(accum_start_field) > 1:
                    raise ValueError(
                        f"Expected a single field for accumulation start at {accum_start_valid_time}, "
                        f"got {len(accum_start_field)}."
                    )

                # if previous field is from previous day
                # then the value is the sum of the current value and the difference between
                # the previous day 24h accumulation and the previous field value
                if accum_start_valid_time < valid_time:
                    LOG.debug("Handling accumulation crossing previous day boundary. valid_time=%s, accum_start_valid_time=%s", valid_time, accum_start_valid_time)
                    day_boundary_valid_time = valid_time.replace(hour=0, minute=0)
                    day_boundary_accum_field = paramgroup.sel(valid_time=datetime.strftime(
                        day_boundary_valid_time,
                        "%Y-%m-%dT%H:%M:%S",
                    ))
                    if len(day_boundary_accum_field) == 0:
                        raise ValueError(
                            f"Cannot compute accumulation for field at {valid_time} and "
                            f"period of {self.accumulation_period_hours}h, missing previous "
                            f"day boundary field at {day_boundary_valid_time}."
                        )
                    accum_field = field.values + (day_boundary_accum_field.values - accum_start_field.values).squeeze()
                    start_step = f"-{encode_step(timedelta(hours=self.accumulation_period_hours))}"
                    md = field.metadata().override(startStep=start_step)
                    accum_fl.append(field.copy(values=accum_field, metadata=md))
                    continue
                
                # in the regular case, the accumulation is simply the difference
                # between the current and the previous field
                accum_field = field.values - accum_field.values
                start_step = f"-{encode_step(timedelta(hours=self.accumulation_period_hours))}"
                md = field.metadata().override(startStep=start_step)
                accum_fl.append(field.copy(values=accum_field, metadata=md))
        return accum_fl


# class _VerticalInterpolationMixin(ReaLCh1Base):
#     """Vertical interpolation mixin for REA-L-CH1 data source."""

#     @property
#     def target_levels(self) -> list[float]:
#         try:
#             return self.mixins_kwargs["target_levels"]
#         except KeyError:
#             raise ValueError("target_levels must be provided for vertical interpolation.")

#     @property
#     def target_levels_units(self) -> str:
#         try:
#             return self.mixins_kwargs["target_levels_units"]
#         except KeyError:
#             raise ValueError("target_levels_units must be provided for vertical interpolation.")

#     def process_data(self, fl: ekd.FieldList) -> ekd.FieldList:
#         fl = super().process_data(fl)
#         out_fl = ekd.SimpleFieldList()
#         for paramgroup in fl.group_by("param"):
#             pressure_fields = fl.sel(param=VARIABLES_PROPERTIES["P"]["paramId"])
#             if len(pressure_fields) == 0:
#                 raise ValueError("Cannot find pressure levels fields required for vertical interpolation.")
#             for field in paramgroup.order_by("valid_time"):
#                 pressure_field = pressure_fields.sel(valid_time=field.metadata("valid_time"))
#                 if len(pressure_field) == 0:
#                     raise ValueError(
#                         f"Cannot find pressure levels field for valid_time "
#                         f"{field.metadata('valid_time')} required for vertical interpolation."
#                     )
#                 interpolated_values = interpolate_k2p_linear_in_lnp_parallel(
#                     field.values,
#                     pressure_field.values,
#                     self.target_levels,
#                     self.target_levels_units,
#                 )
#                 out_fl.append(field.copy(values=interpolated_values))
#         return out_fl
    
    
# class _AggregationAverage(ReaLCh1Base):
#     """Aggregation average variable source."""

#     def __init__(self, context, request, aggregation_period_minutes: int):
#         self.aggregation_period_minutes = aggregation_period_minutes
#         super().__init__(context, request)

#     @staticmethod
#     def time_request(dt: datetime) -> str:
#         """Defines the time-related keys for the FDB request."""
#         out = {}
#         out["date"] = dt.strftime("%Y%m%d")
#         out["time"] = "0000"
#         out["step"] = int((dt - dt.replace(hour=0, minute=0)).total_seconds() // 3600)
#         return out
    
#     def prepare_requests(self, dates: list[datetime]) -> list[dict[str, Any]]:
#         requests = []
#         for date in dates:
#             time_request = self.time_request(date)
#             requests.append(self.request | time_request)
#         return requests


def get_sources(requests: dict[str, Any]) -> ekd.FieldList:
    """Get data from FDB for the given request and dates.
    


    Parameters
    ----------
    requests : dict[str, Any]
        The FDB requests to load.
    
    Returns
    -------
    ekd.FieldList
        The loaded data with adjusted metadata.
    """
    fl = ekd.SimpleFieldList()
    for request in requests:
        fl += ekd.from_source("fdb", request)
    # if len(fl) == 0:
        # raise ValueError(f"No data found for requests: {requests}")
    breakpoint()
    return fl

def override_time_metadata(field: ekd.Field) -> ekd.Field:
    """Override time-related metadata of a field.

    Each field in the returned FieldList will have its metadata modified 
    to have time keys encoded as one would expect for a reanalysis dataset. 
    That is, the data datetime will be set to the validity datetime and the 
    step will be set to 0; the startStep/endStep (important for 
    accumulations/aggregations) will also be adjusted accordingly.

    Parameters
    ----------
    field : ekd.Field
        The input field.

    Returns
    -------
    ekd.Field
        The field with adjusted time metadata.
    """
    md = field.metadata()
    time_keys = {
        "dataDate": md["validityDate"],
        "dataTime": md["validityTime"],
        "step": 0,
    }
    end_step = decode_step(md["endStep"])
    time_keys["endStep"] = 0
    match md["stepType"]:
        case "instant":
            time_keys["startStep"] = 0
        case "accum":
            time_keys["startStep"] = f"-{encode_step(end_step)}"
        case _:
            raise ValueError(f"Unsupported stepType: {md['stepType']}")
    
    md = md.override(time_keys)
    field = field.copy(values=field.values, metadata=md)
    return field



def _interpolate_to_pressure_levels(
        fields: dict[str, xr.DataArray],
        pressure: xr.DataArray,
        p_lev: list[float],
    ) -> dict[str, xr.DataArray]:
    """Interpolate to pressure levels and extrapolate below the surface where needed."""

    pressure[{"z": 0}] = pressure[{"z": 0}].where(
        pressure[{"z": 0}] < 5000, 5000 - 1e-5
    )
    res = {}
    for name, field in fields.items():
        res[name] = vertical_interpolation.interpolate_k2p(field, "linear_in_lnp", pressure, p_lev, "hPa")
    return res