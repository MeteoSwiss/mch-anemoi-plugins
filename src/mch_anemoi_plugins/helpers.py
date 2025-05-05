from copy import copy

import earthkit.data as ekd
import numpy as np
from anemoi.transform.fields import new_field_from_numpy, new_fieldlist_from_list
from anemoi.transform.filter import Filter
from pyproj import CRS, Transformer


def reproject(x_coords, y_coords, src_crs, dst_crs):
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return transformer.transform(x_coords, y_coords)


def replace(instance, **kwargs):
    new_instance = copy(instance)
    for k, v in kwargs.items():
        if hasattr(new_instance, k):
            setattr(new_instance, k, v)
        else:
            raise AttributeError(f"Attribute '{k}' does not exist in the instance.")
    return new_instance


def assign_lonlat(array, crs):
    xv, yv = np.meshgrid(array.x, array.y, indexing="ij")
    lon, lat = reproject(xv, yv, crs, CRS.from_user_input("epsg:4326"))
    return array.assign_coords(longitude=(("x", "y"), lon), latitude=(("x", "y"), lat))


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
