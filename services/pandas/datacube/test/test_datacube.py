from __future__ import absolute_import

from ..datacube import Datacube
import pytest
import numpy as np
import xarray as xr


@pytest.fixture(params=[(20,30), (10,20,30)])
def test_nd_netcdf(request, tmpdir_factory):
    x = np.random.random(request.param)
    dims = ['dim_{0}'.format(i) for i in range(x.ndim)]
    ds = xr.Dataset(data_vars={'foo': (dims, x)})
    nc_file = str(tmpdir_factory.mktemp('data').join('foo.nc'))
    ds.to_netcdf(nc_file, format='NETCDF4')
    return nc_file


def test_test(redisdb):
    d = Datacube()
    assert d.test() == 1


def test___init__(test_nd_netcdf, redisdb):
    d = Datacube(test_nd_netcdf)

