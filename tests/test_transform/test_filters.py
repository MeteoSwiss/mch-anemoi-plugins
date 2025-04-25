import earthkit.data as ekd
from anemoi.transform.filters import filter_registry
from meteodatalab import data_source
from meteodatalab import grib_decoder
from meteodatalab.operators.clip import clip_lateral_boundary_strip
from meteodatalab.operators.destagger import destagger
from numpy.testing import assert_array_equal

from mch_anemoi_plugins.transform.filters import ClipLateralBoundaries
from mch_anemoi_plugins.transform.filters import HorizontalDestagger


def test_clip_lateral_boundaries(data_dir):
    fn = str(data_dir / "kenda-ch1-sfc.grib")
    gridfile_fn = "/scratch/mch/jenkins/icon/pool/data/ICON/mch/grids/icon-1/icon_grid_0001_R19B08_mch.nc"
    strip_idx = 14

    filter: ClipLateralBoundaries = filter_registry.create(
        "clip_lateral_boundaries", strip_idx, gridfile_fn
    )

    # expected
    source = data_source.FileDataSource(datafiles=[fn])
    ds = grib_decoder.load(source, {"param": ["T_2M"]})
    ds["T_2M"] = clip_lateral_boundary_strip(ds["T_2M"], strip_idx)

    # actual
    fieldlist = ekd.from_source("file", fn)
    res = filter.forward(fieldlist)

    assert_array_equal(ds["T_2M"].values.ravel(), res[0].values)


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
