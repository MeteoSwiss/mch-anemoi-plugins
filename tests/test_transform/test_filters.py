import earthkit.data as ekd
from anemoi.transform.filters import filter_registry
from meteodatalab import data_source
from meteodatalab import grib_decoder
from meteodatalab.operators.destagger import destagger
from numpy.testing import assert_array_equal

from mch_anemoi_plugins.transform.filter import HorizontalDestagger


def test_destagger(data_dir):
    fn = str(data_dir / "kenda-1-uv-ml60.grib")
    param_dim = {"U": "x", "V": "y"}

    filter: HorizontalDestagger = filter_registry.create("destagger", param_dim)

    # expected
    source = data_source.FileDataSource(datafiles=[fn])
    ds = grib_decoder.load(source, {"param": ["U", "V"]})
    ds["U"].attrs["origin_x"] = 0.5
    ds["V"].attrs["origin_y"] = 0.5
    ds = {k: destagger(v, param_dim[k]) for k, v in ds.items()}

    # actual
    fieldlist = ekd.from_source("file", fn)
    res = filter.forward(fieldlist)

    assert_array_equal(ds["U"].values.ravel(), res[0].values)
