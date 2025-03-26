"""Plugins for anemoi-transform's filters."""

from anemoi.transform.filter import Filter
from anemoi.transform.fields import new_field_from_numpy
from anemoi.transform.fields import new_fieldlist_from_list
import earthkit.data as ekd
from earthkit.data import Field
import xarray as xr

from meteodatalab.grib_decoder import _FieldBuffer, _is_ensemble
import meteodatalab.operators.destagger as dsg


def _destagger_field(field: Field, dim: str) -> xr.DataArray:
    buffer = _FieldBuffer(_is_ensemble(field))
    buffer.load(field)
    da = buffer.to_xarray()

    # TODO: this hardcoding is not ideal
    if f"origin_{dim}" not in da.attrs:
        da.attrs[f"origin_{dim}"] = 0.5

    return dsg.destagger(da, dim).squeeze(drop=True)


# class HorizontalDestagger(MatchingFieldsFilter):
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


class ExamplePlugin(Filter):
    """A filter to do something on fields."""

    # The version of the plugin API, used to ensure compatibility
    # with the plugin manager.

    api_version = "1.0.0"

    # The schema of the plugin, used to validate the parameters.
    # This is a Pydantic model.

    schema = None

    def __init__(self, factor: float = 2.0):
        """Initialise the filter with the user's parameters"""

        self.factor = factor

    def forward(self, data: ekd.FieldList) -> ekd.FieldList:
        """Multiply all field values by self.factor"""

        result = []
        for field in data:  # Loop over all fields in the input data
            values = (
                field.to_numpy() * self.factor
            )  # Multiply the field values by self.factor

            out = new_field_from_numpy(
                values,
                template=field,  # Use the input field as a template for the output field
                param=field.metadata("param")
                + "_modified",  # Add "_modified" to the parameter name
            )

            result.append(out)

        # Return the modified fields
        return new_fieldlist_from_list(result)
