import pytest

import hierophant as hp


@pytest.mark.parametrize("name", hp.datasets.names("bundled"))
def test_bundled_loads_and_fits(name):
    ds = hp.datasets.load(name)
    assert len(ds.X) > 100 and ds.score() > 0.1


def test_every_entry_has_source_and_license():
    for e in hp.datasets.REGISTRY.values():
        assert e.source and e.license and e.tier in ("bundled", "download")


@pytest.mark.network
@pytest.mark.parametrize("name", hp.datasets.names("download"))
def test_downloaded_loads_and_fits(name):
    ds = hp.datasets.load(name, n_estimators=50)
    assert len(ds.X) > 500 and ds.score() > 0.3
