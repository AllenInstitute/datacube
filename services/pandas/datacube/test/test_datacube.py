from __future__ import absolute_import

from ..datacube import Datacube
import pytest
import numpy as np
import xarray as xr
from six import iteritems
from itertools import product
import json


@pytest.fixture(scope='session', params=[(20,30), (10,20,30)])
def test_nd_netcdf(request, tmpdir_factory):
    np.random.seed(0)
    shape = request.param
    ndim = len(shape)
    dims = ['dim_{0}'.format(i) for i in range(ndim)]
    data_vars = {'foo_{0}'.format(i): (dims[:(i+1)], np.random.random(shape[:(i+1)])) for i in range(ndim)}
    ds = xr.Dataset(data_vars=data_vars)
    nc_file = str(tmpdir_factory.mktemp('data').join('foo.nc'))
    ds.to_netcdf(nc_file, format='NETCDF4')
    return nc_file, ds


@pytest.fixture(params=list(product((False, True), (8*10, 8*200))))
def test_datacube(request, test_nd_netcdf, redisdb):
    use_chunks, max_cacheable_bytes = request.param
    nc_file, ds = test_nd_netcdf
    chunks = None
    if use_chunks:
        chunks = {dim: 5 for dim in ds.dims}
    d = Datacube('test', nc_file, chunks=chunks, max_cacheable_bytes=max_cacheable_bytes)
    return d, ds


def test_test(test_datacube):
    d, _ = test_datacube
    assert d.test() == 1


def test___init__(test_datacube):
    d, _ = test_datacube


def test_raw(test_datacube):
    d, ds = test_datacube
    assert d.raw().equals(ds)


def test_raw_max_response_size(test_nd_netcdf, redisdb):
    with pytest.raises(ValueError):
        d = Datacube('test', test_nd_netcdf[0], max_response_size=8*299)
        d.raw(select={'dim_0': {'start': 0, 'stop': 10}})


def test_raw_select(test_datacube):
    d, ds = test_datacube
    r = d.raw(select={'dim_0': {'start': 0, 'stop': 10}})
    assert r.equals(ds[{'dim_0': slice(0,10)}])
    r = d.raw(select={'dim_0': [0,2,5], 'dim_1': [3,5,7,9]})
    assert r.equals(ds[{'dim_0': [0,2,5], 'dim_1': [3,5,7,9]}])


#todo: should be able to use Dataset/DataArray equals() but there seems to be a bug with data not having the right shape when a
# dimension has size 0. using Dataset/DataArray to_dict() as a workaround for now
def test_raw_filters(test_datacube):
    d, ds = test_datacube
    r = d.raw(filters=[{'field': 'foo_0', 'op': '<=', 'value': 0.5}])
    #assert r.equals(ds.where(ds.foo_0 <= 0.5, drop=True))
    assert json.dumps(r.to_dict()) == json.dumps(ds.where(ds.foo_0 <= 0.5, drop=True).to_dict())
    r = d.raw(filters={'and': [{'field': 'foo_0', 'op': '<=', 'value': 0.25},{'field': 'foo_0', 'op': '>=', 'value': 0.75}]})
    #assert r.equals(ds.where((ds.foo_0 <= 0.25) & (ds.foo_0 >= 0.75), drop=True))
    assert json.dumps(r.to_dict()) == json.dumps(ds.where((ds.foo_0 <= 0.25) & (ds.foo_0 >= 0.75), drop=True).to_dict())
    r = d.raw(filters={'or': [{'field': 'foo_0', 'op': '<=', 'value': 0.25},{'field': 'foo_0', 'op': '>=', 'value': 0.75}]})
    #assert r.equals(ds.where((ds.foo_0 <= 0.25) | (ds.foo_0 >= 0.75), drop=True))
    assert json.dumps(r.to_dict()) == json.dumps(ds.where((ds.foo_0 <= 0.25) | (ds.foo_0 >= 0.75), drop=True).to_dict())
    r = d.raw(filters={'and': [{'field': 'foo_0', 'op': '<=', 'value': 0.5},{'field': 'foo_1', 'op': '<=', 'value': 0.5}]})
    cond = (ds.foo_0 <= 0.5) & (ds.foo_1 <= 0.5)
    #assert r.foo_0.equals(ds.where(cond.any(dim='dim_1'), drop=True).foo_0)
    assert json.dumps(r.foo_0.to_dict()) == json.dumps(ds.where(cond.any(dim='dim_1'), drop=True).foo_0.to_dict())
    #assert r.foo_1.equals(ds.where(cond, drop=True).foo_1)
    assert json.dumps(r.foo_1.to_dict()) == json.dumps(ds.where(cond, drop=True).foo_1.to_dict())
    #if 'foo_2' in ds: assert r.foo_2.equals(ds.where(cond, drop=True).foo_2)
    if 'foo_2' in ds: assert json.dumps(r.foo_2.to_dict()) == json.dumps(ds.where(cond, drop=True).foo_2.to_dict())
    r = d.raw(filters={'or': [{'field': 'foo_0', 'op': '<=', 'value': 0.25},{'field': 'foo_1', 'op': '<=', 'value': 0.1}]})
    cond = (ds.foo_0 <= 0.25) | (ds.foo_1 <= 0.1)
    #assert r.foo_0.equals(ds.where(cond.any(dim='dim_1'), drop=True).foo_0)
    assert json.dumps(r.foo_0.to_dict()) == json.dumps(ds.where(cond.any(dim='dim_1'), drop=True).foo_0.to_dict())
    #assert r.foo_1.equals(ds.where(cond, drop=True).foo_1)
    assert json.dumps(r.foo_1.to_dict()) == json.dumps(ds.where(cond, drop=True).foo_1.to_dict())
    #if 'foo_2' in ds: assert r.foo_2.equals(ds.where(cond, drop=True).foo_2)
    if 'foo_2' in ds: assert json.dumps(r.foo_2.to_dict()) == json.dumps(ds.where(cond, drop=True).foo_2.to_dict())
