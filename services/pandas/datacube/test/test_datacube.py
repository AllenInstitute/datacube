from __future__ import absolute_import

from ..datacube import Datacube
import pytest
import numpy as np
import xarray as xr
from six import iteritems
from itertools import product
import json


#todo: add tests with nan's
@pytest.fixture(scope='session', params=[(10,20), (10,20,30)])
def test_nd_netcdf(request, tmpdir_factory):
    np.random.seed(0)
    shape = request.param
    ndim = len(shape)
    dims = ['dim_{0}'.format(i) for i in range(ndim)]
    coords = {'dim_{0}'.format(i): range(shape[i]) for i in range(ndim)}
    data_vars = {'foo_{0}'.format(i): (dims[:(i+1)], np.random.random(shape[:(i+1)])) for i in range(ndim)}
    ds = xr.Dataset(coords=coords, data_vars=data_vars)
    nc_file = str(tmpdir_factory.mktemp('data').join('foo.nc'))
    ds.to_netcdf(nc_file, format='NETCDF4', engine='h5netcdf')
    return nc_file, ds


@pytest.fixture(params=list(product((False,), (8*10, 8*200))))
def test_datacube(request, test_nd_netcdf, redisdb):
    use_chunks, max_cacheable_bytes = request.param
    nc_file, ds = test_nd_netcdf
    chunks = None
    if use_chunks:
        chunks = {dim: 3 for dim in ds.dims}
    d = Datacube('test', nc_file, redis_client=redisdb, chunks=chunks, max_cacheable_bytes=max_cacheable_bytes, num_chunks=1, max_workers=1, recache=True)
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
        d = Datacube('test', test_nd_netcdf[0], max_response_size=8*199, redis_client=redisdb, recache=True)
        d.raw(select={'dim_0': {'start': 0, 'stop': 10}})


@pytest.mark.filterwarnings('ignore')
def test_raw_select(test_datacube):
    d, ds = test_datacube
    r = d.raw(select={'dim_0': {'start': 0, 'stop': 8}})
    assert r.equals(ds[{'dim_0': slice(0,8)}])
    r = d.raw(select={'dim_0': [0,2,5], 'dim_1': [3,5,7,9]})
    assert r.equals(ds[{'dim_0': [0,2,5], 'dim_1': [3,5,7,9]}])


@pytest.mark.filterwarnings('ignore')
def test_raw_filters(test_datacube):
    d, ds = test_datacube

    r = d.raw(filters=[{'field': 'foo_0', 'op': '<=', 'value': 0.5}])
    assert r.equals(ds.where(ds.foo_0 <= 0.5, drop=True))

    r = d.raw(filters={'and': [{'field': 'foo_0', 'op': '<=', 'value': 0.25},{'field': 'foo_0', 'op': '>=', 'value': 0.75}]})
    assert r.equals(ds.where((ds.foo_0 <= 0.25) & (ds.foo_0 >= 0.75), drop=True))

    r = d.raw(filters={'or': [{'field': 'foo_0', 'op': '<=', 'value': 0.25},{'field': 'foo_0', 'op': '>=', 'value': 0.75}]})
    assert r.equals(ds.where((ds.foo_0 <= 0.25) | (ds.foo_0 >= 0.75), drop=True))

    r = d.raw(filters={'and': [{'field': 'foo_0', 'op': '<=', 'value': 0.5},{'field': 'foo_1', 'op': '<=', 'value': 0.5}]})
    cond = (ds.foo_0 <= 0.5) & (ds.foo_1 <= 0.5)
    assert r.foo_0.equals(ds.where(cond.any(dim='dim_1'), drop=True).foo_0)
    assert r.foo_1.equals(ds.where(cond, drop=True).foo_1)
    if 'foo_2' in ds: assert r.foo_2.equals(ds.where(cond, drop=True).foo_2)

    r = d.raw(filters={'or': [{'field': 'foo_0', 'op': '<=', 'value': 0.25},{'field': 'foo_1', 'op': '<=', 'value': 0.1}]})
    cond = (ds.foo_0 <= 0.25) | (ds.foo_1 <= 0.1)
    assert r.foo_0.equals(ds.where(cond.any(dim='dim_1'), drop=True).foo_0)
    assert r.foo_1.equals(ds.where(cond, drop=True).foo_1)
    if 'foo_2' in ds: assert r.foo_2.equals(ds.where(cond, drop=True).foo_2)


@pytest.mark.filterwarnings('ignore')
def test_raw_filters_coords(test_datacube):
    d, ds = test_datacube

    r = d.raw(filters={'or': [{'field': 'foo_0', 'op': '<=', 'value': 0.25},{'field': 'foo_1', 'coords': {'dim_0': 5}, 'op': '<=', 'value': 0.1}]})
    cond = (ds.foo_0 <= 0.25) | (ds.foo_1.loc[dict(dim_0=5)] <= 0.1)
    assert r.foo_0.equals(ds.where(cond.any(dim='dim_1'), drop=True).foo_0)
    assert r.foo_1.equals(ds.where(cond, drop=True).foo_1)
    if 'foo_2' in ds: assert r.foo_2.equals(ds.where(cond, drop=True).foo_2)

    r = d.raw(filters={'or': [{'field': 'foo_0', 'op': '<=', 'value': 0.25},{'field': 'foo_1', 'coords': {'dim_1': 7}, 'op': '<=', 'value': 0.1}]})
    cond = (ds.foo_0 <= 0.25) | (ds.foo_1.loc[dict(dim_1=7)] <= 0.1)
    assert r.foo_0.equals(ds.where(cond, drop=True).foo_0)
    assert r.foo_1.equals(ds.where(cond, drop=True).foo_1)
    if 'foo_2' in ds: assert r.foo_2.equals(ds.where(cond, drop=True).foo_2)


@pytest.mark.filterwarnings('ignore')
def test_raw_coords_with_filter_coords(test_datacube):
    d, ds = test_datacube
    r = d.raw(coords={'dim_1': 5}, filters=[{'field': 'foo_1', 'coords': {'dim_1': 4}, 'op': '>=', 'value': .5}])
    cond = ds.foo_1.sel(dim_1=4, drop=True) >= .5
    assert r.equals(ds.sel(dim_1=5).where(cond, drop=True))


@pytest.mark.filterwarnings('ignore')
def test_raw_distance_filter(test_datacube):
    d, ds = test_datacube

    r = d.raw(filters=[{'op': 'distance', 'fields': ['foo_0'], 'point': [.5], 'value': .166}])
    cond = ((ds.foo_0-.5)**2.)**.5<=.166
    assert r.foo_0.equals(ds.where(cond, drop=True).foo_0)
    assert r.foo_1.equals(ds.where(cond, drop=True).foo_1)
    if 'foo_2' in ds: assert r.foo_2.equals(ds.where(cond, drop=True).foo_2)

    r = d.raw(filters=[{'op': 'distance', 'fields': ['dim_0'], 'point': [5.], 'value': 3.}])
    cond = ((ds.dim_0-5.)**2.)**.5<=3.
    assert r.foo_0.equals(ds.where(cond, drop=True).foo_0)
    assert r.foo_1.equals(ds.where(cond, drop=True).foo_1)
    if 'foo_2' in ds: assert r.foo_2.equals(ds.where(cond, drop=True).foo_2)

    r = d.raw(filters=[{'op': 'distance', 'fields': ['dim_0', 'dim_0'], 'point': [5.,5.], 'value': 3.}])
    cond = ((ds.dim_0-5.)**2.+(ds.dim_0-5.)**2.)**.5<=3.
    assert r.foo_0.equals(ds.where(cond, drop=True).foo_0)
    assert r.foo_1.equals(ds.where(cond, drop=True).foo_1)
    if 'foo_2' in ds: assert r.foo_2.equals(ds.where(cond, drop=True).foo_2)

    r = d.raw(filters=[{'op': 'distance', 'fields': ['dim_0', 'dim_0', 'dim_0'], 'point': [5.,5.,5.], 'value': 3.}])
    cond = ((ds.dim_0-5.)**2.+(ds.dim_0-5.)**2.+(ds.dim_0-5.)**2.)**.5<=3.
    assert r.foo_0.equals(ds.where(cond, drop=True).foo_0)
    assert r.foo_1.equals(ds.where(cond, drop=True).foo_1)
    if 'foo_2' in ds: assert r.foo_2.equals(ds.where(cond, drop=True).foo_2)

    r = d.raw(filters=[{'op': 'distance', 'fields': ['dim_0', 'dim_1'], 'point': [5.,10.], 'value': 3.}])
    cond = ((ds.dim_0-5.)**2.+(ds.dim_1-10.)**2.)**.5<=3.
    assert r.foo_0.equals(ds.where(cond.any(dim='dim_1'), drop=True).foo_0)
    assert r.foo_1.equals(ds.where(cond, drop=True).foo_1)
    if 'foo_2' in ds: assert r.foo_2.equals(ds.where(cond, drop=True).foo_2)

    r = d.raw(filters=[{'op': 'distance', 'fields': ['foo_0', {'field': 'foo_1', 'coords': {'dim_1': 6}}], 'point': [.5,.3], 'value': .166}])
    cond = ((ds.foo_0-.5)**2.+(ds.foo_1.sel(dim_1=6, drop=True)-.3)**2.)**.5<=.166
    assert r.foo_0.equals(ds.where(cond, drop=True).foo_0)
    assert r.foo_1.equals(ds.where(cond, drop=True).foo_1)
    if 'foo_2' in ds: assert r.foo_2.equals(ds.where(cond, drop=True).foo_2)

    r = d.raw(filters=[{'op': 'distance', 'fields': ['foo_0', {'field': 'foo_1', 'coords': {'dim_0': 7}}], 'point': [.5,.3], 'value': .166}])
    cond = ((ds.foo_0-.5)**2.+(ds.foo_1.sel(dim_0=7, drop=True)-.3)**2.)**.5<=.166
    assert r.foo_0.equals(ds.where(cond.any(dim='dim_1'), drop=True).foo_0)
    assert r.foo_1.equals(ds.where(cond, drop=True).foo_1)
    if 'foo_2' in ds: assert r.foo_2.equals(ds.where(cond, drop=True).foo_2)

    if 'foo_2' in ds:
        r = d.raw(filters=[{'op': 'distance', 'fields': ['dim_0', 'dim_1', 'dim_2'], 'point': [5.,10.,15.], 'value': 3.}])
        cond = ((ds.dim_0-5.)**2.+(ds.dim_1-10.)**2.+(ds.dim_2-15.)**2.)**.5<=3.
        assert r.foo_0.equals(ds.where(cond.any(dim=['dim_1', 'dim_2']), drop=True).foo_0)
        assert r.foo_1.equals(ds.where(cond.any(dim='dim_2'), drop=True).foo_1)
        assert r.foo_2.equals(ds.where(cond, drop=True).foo_2)


def pearsonr(x, y):
    mx = np.ma.array(x, mask=np.isnan(x))
    my = np.ma.array(y, mask=np.isnan(y))
    from scipy.stats.mstats import pearsonr as mstats_pearsonr
    return np.array(mstats_pearsonr(mx,my)[0])


def corr(ds, field, dim, label):
    seed = ds[field].sel(**{dim: label})
    rvalues = [pearsonr(seed, ds[field].sel(**{dim: i})) for i in ds[dim]]
    corr = xr.Dataset({'corr': ([dim], rvalues), dim: ds[dim]})
    sample_dims = set(ds[field].dims)-set([dim])
    both_samples_count = xr.ufuncs.logical_and(~np.isnan(seed), ~np.isnan(ds[field])).sum(dim=sample_dims)
    corr = corr.where(both_samples_count>1, drop=True)
    return corr.dropna(dim, how='all', subset=['corr'])


@pytest.mark.filterwarnings('ignore')
def test_corr(test_datacube):
    d, ds = test_datacube

    r = d.corr('foo_0', 'dim_0', 0)
    expected = corr(ds, 'foo_0', 'dim_0', 0)
    xr.testing.assert_allclose(r, expected)

    with pytest.raises(KeyError):
        r = d.corr('foo_0', 'dim_0', -1)

    r = d.corr('foo_1', 'dim_0', 0)
    expected = corr(ds, 'foo_1', 'dim_0', 0)
    xr.testing.assert_allclose(r, expected)

    r = d.corr('foo_1', 'dim_1', 0)
    expected = corr(ds, 'foo_1', 'dim_1', 0)
    xr.testing.assert_allclose(r, expected)

    if 'foo_2' in ds:
        r = d.corr('foo_2', 'dim_0', 0)
        expected = corr(ds, 'foo_2', 'dim_0', 0)
        xr.testing.assert_allclose(r, expected)

        r = d.corr('foo_2', 'dim_1', 0)
        expected = corr(ds, 'foo_2', 'dim_1', 0)
        xr.testing.assert_allclose(r, expected)

        r = d.corr('foo_2', 'dim_2', 0)
        expected = corr(ds, 'foo_2', 'dim_2', 0)
        xr.testing.assert_allclose(r, expected)


@pytest.mark.filterwarnings('ignore')
def test_corr_select(test_datacube):
    d, ds = test_datacube

    r = d.corr('foo_1', 'dim_0', 0, select={'dim_1': {'start': 0, 'stop': 8}})
    s = ds[{'dim_1': slice(0,8)}]
    expected = corr(s, 'foo_1', 'dim_0', 0)
    xr.testing.assert_allclose(r, expected)

    r = d.corr('foo_1', 'dim_0', 0, select={'dim_0': [0,2,5], 'dim_1': [3,5,7,9]})
    s = ds[{'dim_0': [0,2,5], 'dim_1': [3,5,7,9]}]
    expected = corr(s, 'foo_1', 'dim_0', 0)
    xr.testing.assert_allclose(r, expected)


#todo: make sure and test when mask has different dimensions than data
# (will need to add an option to skip the nan-masking in order to do this)
@pytest.mark.filterwarnings('ignore')
def test_corr_filters(test_datacube):
    d, ds = test_datacube

    #fixme: empty filter result
    #r = d.corr('foo_1', 'dim_0', 0, filters={'or': [{'field': 'foo_0', 'op': '<=', 'value': 0.}]})
    #cond = ds.foo_0 < 0.
    #s = ds.where(cond, drop=True)
    #xr.testing.assert_allclose(r, xr.Dataset({'corr': (['dim_0'], np.array([pearsonr(s.foo_1.sel(dim_0=0), s.foo_1.isel(dim_0=i)) for i in range(s.dims['dim_0'])])), 'dim_0': range(ds.dims['dim_0'])}))

    r = d.corr('foo_1', 'dim_0', 0, filters={'or': [{'field': 'foo_0', 'op': '<=', 'value': 0.75}]})
    cond = ds.foo_0 <= 0.75
    s = ds.where(cond, drop=True)
    expected = corr(s, 'foo_1', 'dim_0', 0)
    xr.testing.assert_allclose(r, expected)

    r = d.corr('foo_1', 'dim_0', 0, filters={'or': [{'field': 'foo_0', 'op': '<=', 'value': 0.25},{'field': 'foo_1', 'op': '<=', 'value': 0.1}]})
    cond = (ds.foo_0 <= 0.25) | (ds.foo_1 <= 0.1)
    s = ds.where(cond, drop=True)
    expected = corr(s, 'foo_1', 'dim_0', 0)
    xr.testing.assert_allclose(r, expected)

    r = d.corr('foo_1', 'dim_0', 0, filters={'or': [{'field': 'foo_1', 'op': '<=', 'value': 0.5}]})
    cond = ds.foo_1 <= 0.5
    s = ds.where(cond, drop=True)
    expected = corr(s, 'foo_1', 'dim_0', 0)
    xr.testing.assert_allclose(r, expected)

    if 'foo_2' in ds:
        r = d.corr('foo_1', 'dim_0', 0, filters={'or': [{'field': 'foo_2', 'op': '<=', 'value': 0.01}]})
        cond = ds.foo_2 <= 0.01
        s = ds.where(cond.any(dim='dim_2'), drop=True)
        expected = corr(s, 'foo_1', 'dim_0', 0)
        xr.testing.assert_allclose(r, expected)

        r = d.corr('foo_2', 'dim_2', 0, filters={'or': [{'field': 'foo_1', 'op': '<=', 'value': 0.1}]})
        cond = ds.foo_1 <= 0.1
        s = ds.where(cond, drop=True)
        expected = corr(s, 'foo_2', 'dim_2', 0)
        xr.testing.assert_allclose(r, expected)

        r = d.corr('foo_2', 'dim_1', 0, filters={'or': [{'field': 'foo_1', 'op': '<=', 'value': 0.1},{'field': 'dim_1', 'op': '=', 'value': 0}]})
        cond = (ds.foo_1 <= 0.1) | (ds.dim_1 == 0)
        s = ds.where(cond, drop=True)
        expected = corr(s, 'foo_2', 'dim_1', 0)
        xr.testing.assert_allclose(r, expected)

        with pytest.raises(KeyError):
            r = d.corr('foo_2', 'dim_1', -1, filters={'or': [{'field': 'foo_1', 'op': '<=', 'value': 0.1}]})
