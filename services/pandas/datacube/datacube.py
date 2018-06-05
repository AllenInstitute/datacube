import numpy as np
from scipy.stats import rankdata
import json
import pickle
import redis
#import txredisapi
#from twisted.internet import reactor
import xarray as xr
import xarray.ufuncs as xr_ufuncs
import zarr
import numcodecs
import dask
import dask.array as da
import multiprocessing
from PIL import Image
from io import BytesIO
import os
import shutil
import base64
import itertools
import functools
from functools import reduce, wraps
import operator
import inspect
import types
import concurrent.futures
import collections
from twisted.python import log
import logging

from six import iteritems
from builtins import int, map

from . import summary_statistics


def get_num_samples(data, axis, masks, backend=np):
    sample_axes = tuple([i for i in range(data.ndim) if i != axis])

    if not masks or all(mask.ndim == 0 for mask in masks):
        num_samples = np.prod(np.array([data.shape[i] for i in sample_axes], dtype=np.uint64))
        if backend == da:
            num_samples = da.from_array(num_samples)
    elif len(masks) == 1 and masks[0].ndim == 0 and not masks[0]:
        num_samples = np.array(0, np.uint64)
    else:
        ddims = range(data.ndim)
        output_shape = tuple(max(mask.shape[i] for mask in masks) if i == axis else 1 for i in ddims)
        mdata_args = list(it.__next__() for it in itertools.cycle([iter(masks), iter(range(mask.ndim) for mask in masks)]))
        num_samples = backend.einsum(*mdata_args, [axis], dtype=np.uint64)
        num_samples = backend.reshape(num_samples, output_shape) # restore dims after einsum
        if len(sample_axes)>0:
            num_samples *= np.prod(np.array([data.shape[i] for i in sample_axes
                if all(mask.shape[i]==1 for mask in masks) and data.shape[i]>1], dtype=np.uint64))
    return num_samples


def einsum_corr(data, seed, axis, masks, mseed, backend=np, memoize=lambda k,f,*a,**kw: f(*a,**kw)):
    np.seterr(divide='ignore', invalid='ignore')
    dtype = np.result_type(np.float64, data.dtype, *[mask.dtype for mask in masks], seed.dtype, mseed.dtype) # use f64 at least
    
    ddims = range(data.ndim)
    sdims = range(seed.ndim)
    msdims = range(mseed.ndim)
    mdata_args = list(it.__next__() for it in itertools.cycle([iter(masks), iter(range(mask.ndim) for mask in masks)]))
    sample_axes = tuple([i for i in ddims if i != axis])
    output_shape = tuple(data.shape[i] if i == axis else 1 for i in ddims)

    seed_num_samples = get_num_samples(seed, axis, [mseed], backend=backend)
    num_samples = memoize(['num_samples'], get_num_samples, data, axis, masks, backend=backend)
    both_samples = get_num_samples(data, axis, masks+[mseed], backend=backend)
    mseed = mseed.astype(data.dtype)
    for i, mask in enumerate(masks):
        if mask.size<=mseed.size:
            masks[i] = mask.astype(data.dtype)

    seed_mean = backend.einsum(seed, sdims, mseed, msdims, [], dtype=dtype) / seed_num_samples
    data_sum = memoize(['data_sum'], backend.einsum, data, ddims, *mdata_args, [axis], dtype=dtype)
    data_sum = backend.reshape(data_sum, output_shape) # restore dims after einsum
    data_mean = data_sum / num_samples

    seed_dev = backend.einsum(seed-seed_mean, sdims, mseed, msdims, sdims, dtype=dtype)
    seed_denominator = backend.sum(seed_dev**2, axis=sample_axes, dtype=dtype)
    numerator = backend.einsum(seed_dev, ddims, data, ddims, *mdata_args, [axis], dtype=dtype)
    numerator -= backend.einsum(seed_dev, ddims, data_mean, ddims, *mdata_args, [axis], dtype=dtype)
    numerator = backend.reshape(numerator, output_shape) # restore dims after einsum

    data_denominator = memoize(['data_denominator'], backend.einsum, data, ddims, data, ddims, *mdata_args, [axis], dtype=dtype)
    data_denominator = backend.reshape(data_denominator, output_shape) # restore dims after einsum
    data_denominator = data_denominator+data_mean**2*num_samples
    data_denominator = data_denominator-2.0*data_mean*data_sum

    denominator = np.sqrt(seed_denominator*data_denominator)

    corr = backend.clip(numerator / denominator, -1.0, 1.0)
    corr = corr * backend.sign(np.inf*(both_samples>1))
    corr = backend.reshape(corr, tuple(data.shape[i] if i == axis else 1 for i in range(data.ndim)))
    return corr


def get_chunk_inds(ds, dim, num_chunks, chunk_idx):
    chunk_size = 1 + ds.dims[dim] // num_chunks
    return {dim: slice(chunk_idx * chunk_size, min((chunk_idx + 1) * chunk_size, ds.dims[dim]))}


def get_chunk(ds, *args):
    return ds[get_chunk_inds(ds, *args)]


def parallelize(dim_arg, num_chunks_arg, max_workers_arg, exclude_args=[]):
    def parallelize_decorator(f):
        @wraps(f)
        def parallel(*args, **kwargs):
            argspec = inspect.getargspec(f)
            args = list(args)

            dim = args[argspec.args.index(dim_arg)]
            num_chunks = args[argspec.args.index(num_chunks_arg)]
            max_workers = args[argspec.args.index(max_workers_arg)]

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                shape = None
                for chunk_idx in range(0, num_chunks):
                    chunk_args = [get_chunk(arg, dim, num_chunks, chunk_idx)
                                  if isinstance(arg, xr.Dataset)
                                      and argspec[i] not in exclude_args
                                  else arg 
                                  for i, arg in enumerate(args)]
                    futures.append(executor.submit(f, *chunk_args, **{**kwargs, 'chunk_idx': chunk_idx}))

                chunk_results = [None] * num_chunks
                for future in concurrent.futures.wait(futures).done:
                    chunk_result = future.result()
                    chunk_idx = futures.index(future)
                    chunk_results[chunk_idx] = chunk_result

                result = xr.concat(chunk_results, dim=dim)
                return result
        return parallel
    return parallelize_decorator


def get_masks(ds):
    return [ds[v] for v in ds.data_vars if 'is_mask' in ds[v].attrs and ds[v].attrs['is_mask']]


def mask_field(ds, field):
    masks = get_masks(ds)
    if len(masks)>1:
        mask = reduce(xr.ufuncs.logical_and, masks)
    else:
        mask = masks[0]
    return align_mask(ds[field], mask)


def align_mask(data, mask):
    reduce_dims = set(mask.dims)-set(data.dims)
    if reduce_dims:
        mask = mask.any(dim=reduce_dims)
    if data.chunks is not None:
        mask = mask.chunk(tuple(data.chunks[data.dims.index(dim)] for dim in mask.dims))
    expand_dims = set(data.dims)-set(mask.dims)
    if expand_dims:
        mask = mask.expand_dims(expand_dims)
    if data.dims != mask.dims:
        mask = mask.transpose(*data.dims)
    return mask


def get_seed(ds, seed_label, dim):
    data = ds['data']
    seed_ds = ds.sel(**{dim: [seed_label]})
    seed = seed_ds['data']
    mseed = mask_field(seed_ds, 'data')
    return seed, mseed


@parallelize('dim', 'num_chunks', 'max_workers')
def do_correlation(ds, seed, mseed, dim, num_chunks, max_workers, chunk_idx, memoize=lambda k,f,*a,**kw: f(*a,**kw)):
    data = ds['data']
    all_masks = get_masks(ds)
    row_masks = []
    masks = []
    for mask in all_masks:
        if mask.dims==(dim,):
            row_masks.append(mask)
        else:
            masks.append(align_mask(data, mask))
    axis = data.dims.index(dim)
    _memoize = lambda key, f, *args, **kwargs: memoize([chunk_idx]+key, f, *args, **kwargs)
    corr = xr.Dataset({'corr': (data.dims, einsum_corr(data.data, seed, axis, masks, mseed, backend=np, memoize=_memoize))}).squeeze()
    corr.coords[dim] = data.coords[dim]
    if row_masks:
        mask = reduce(xr_ufuncs.logical_and, row_masks)
        corr = corr.reindex_like(mask).where(mask, drop=True)
    return corr


def par_correlation(ds, seed_label, dim, num_chunks, max_workers, memoize=lambda k,f,*a,**kw: f(*a,**kw)):
    seed, mseed = get_seed(ds, seed_label, dim)
    seed = seed.compute()
    mseed = mseed.compute()
    _memoize = lambda key, f, *args, **kwargs: memoize([dim]+key, f, *args, **kwargs)
    corr = do_correlation(ds, seed, mseed, dim, num_chunks, max_workers, memoize=_memoize)
    return corr


# this version uses dask; should add a configuration option to choose
#def par_correlation(ds, seed_label, dim, num_chunks, max_workers, memoize=lambda k,f,*a,**kw: f(*a,**kw)):
#    data = ds['data']
#    mdata = mask_field(ds, 'data')
#    axis = data.dims.index(dim)
#    seed, mseed = get_seed(ds, seed_label, dim)
#    def _compute(arr):
#        if isinstance(arr, da.Array):
#            tmp = arr.compute()
#            return da.from_array(tmp, chunks=arr.chunks)
#        else:
#            return arr
#    _memoize = lambda key, f, *args, **kwargs: memoize([dim]+key, lambda: _compute(f(*args, **kwargs)))
#    corr = einsum_corr(data.data, seed.data, axis, [mdata.data], mseed.data, backend=da, memoize=_memoize)
#    corr = xr.Dataset({'corr': (data.dims, corr)}).squeeze()
#    corr.coords[dim] = ds.coords[dim]
#    return corr


class Datacube:    

    def test(self):
        return 1


    def __init__(self,
                 name,
                 path,
                 session_name=None,
                 redis_client=None,
                 missing_data=False,
                 calculate_stats=True,
                 chunks=None,
                 max_response_size=100*1024*1024,
                 max_cacheable_bytes=100*1024*1024,
                 num_chunks=multiprocessing.cpu_count(),
                 max_workers=multiprocessing.cpu_count(),
                 persist=[]):

        self.name = name
        self.session_name = session_name

        self.max_response_size = max_response_size
        self.max_cacheable_bytes = max_cacheable_bytes
        self.num_chunks = num_chunks
        self.max_workers = max_workers
        if path: 
            self.load(path, chunks, missing_data, calculate_stats, persist)
        #todo: would be nice to find a way to swap these out,
        # and also to be able to run without redis (numpy-only)
        #if reactor.running:
        #    self.redis_client = txredisapi.Connection('localhost', 6379)
        #else:
        if redis_client:
            self.redis_client = redis_client
        else:
            self.redis_client = redis.StrictRedis('localhost', 6379)


    def load_zarr_lmdb(self, path, chunks, exclude=[]):
        ''' Load zarr data lazily from a lightning db store on the filesystem. Uses the in-memory 
        filesystem at /dev/shm for performance.
        '''

        shm_prefix = '/dev/shm'
        if self.session_name is not None:
            shm_prefix = os.path.join(shm_prefix, self.session_name)
        shm_path = os.path.join(shm_prefix, os.path.basename(path))

        log.msg('deleting \'{}\'...'.format(shm_path), logLevel=logging.INFO)
        try:
            shutil.rmtree(shm_path)
        except OSError:
            pass

        try:
            os.makedirs(shm_prefix)
        except OSError:
            pass

        disk_store = zarr.storage.LMDBStore(path, readonly=True)
        shm_store = zarr.storage.LMDBStore(shm_path)

        log.msg('cloning \'{}\' store to \'{}\'...'.format(path, shm_path), logLevel=logging.INFO)
        disk_group = zarr.hierarchy.open_group(disk_store)
        shm_group = zarr.hierarchy.open_group(shm_store)
        for field in set(disk_group)-set(exclude):
            zarr.convenience.copy(disk_group[field], shm_group, name=field)
        disk_store.close()

        log.msg('loading \'{}\' zarr LMDBstore as xarray dataset...'.format(shm_path), logLevel=logging.INFO)
        self.df = xr.open_zarr(store=shm_store, auto_chunk=True)
        self.df = self.df.chunk(chunks)


    def load_nc_file(self, path, chunks):
        print('loading \'{}\' NETCDF4 file as xarray dataset...'.format(path))

        self.df = xr.open_dataset(path, chunks=chunks, engine='h5netcdf')
        for field in self.df.variables:

            if self.df[field].dtype.name.startswith('bytes'):
                self.df[field] = self.df[field].astype(str)


    def load(self, path, chunks=None, missing_data=False, calculate_stats=True, persist=[]):

        #todo: rename df
        #todo: argsorts need to be cached to a file (?)
        if path.endswith('.nc'):
            self.load_nc_file(path, chunks)
        elif path.endswith('.zarr.lmdb'):
            self.load_zarr_lmdb(path, chunks, exclude=persist)
        else:
            raise ValueError('invalid file type; expected *.nc or *.zarr.lmdb')

        #todo: rework _query so this is not needed:
        for dim in self.df.dims:
            if dim not in self.df.dims:
                self.df.coords[dim] = range(self.df.dims[dim])

        if missing_data:
            print('setting up missing data...')
            #todo: need to reintroduce this and test
            #self.mdata = xr.ufuncs.logical_not(xr.ufuncs.isnan(self.df))
            #self.mdata = self.mdata.persist()
            #todo: can nan-only chunks be dropped?
            self.df = self.df.fillna(0.)

        print('building indexes...')
        self.argsorts = {}
        for field in self.df.variables:
            if self.df[field].size*np.dtype('int64').itemsize <= self.max_cacheable_bytes:
                print('building index for field \'{}\'...'.format(field))
                self.argsorts[field] = np.argsort(self.df[field].values, axis=None)

        stats_path = os.path.join(os.path.dirname(path), 'summary_statistics_{}.json'.format(self.name))
        self.stats = summary_statistics.cache_summary_statistics(
            self.df,
            reader=functools.partial(summary_statistics.read_stats_from_json, stats_path),
            writer=functools.partial(summary_statistics.write_stats_to_json, stats_path),
            force=calculate_stats
        )

        if persist:
            if path.endswith('.nc'):
                df = self.df
            elif path.endswith('.zarr.lmdb'):
                disk_store = zarr.storage.LMDBStore(path, readonly=True)
                df = xr.open_zarr(store=disk_store, auto_chunk=True)

            for field in persist:
                print('loading field \'{}\' into memory as ndarray...'.format(field))
                self.df[field] = df[field].load()
        print('done loading \'{}\'.'.format(self.name))


    def _validate_select(self, select):
        assert(all(f in list(self.df.dims.keys()) for f in select.keys()))

        for axis, selector in iteritems(select):
            if isinstance(selector, list):
                if any(type(x) != type(selector[0]) for x in selector):
                    raise RuntimeError('All elements of selector for axis {0} do not have the same type.'.format(axis))
                if not isinstance(selector[0], int) and not isinstance(selector[0], bool):
                    raise RuntimeError('Elements of list selector for axis {0} must be of type int or bool.'.format(axis))
                if isinstance(selector[0], bool) and len(selector) != self.df.dims[axis]:
                    raise RuntimeError('Boolean list selector for axis {0} must have length {1} to match the size of the datacube.'.format(axis, self.df.dims[axis]))
            elif isinstance(selector, dict):
                keys = ['start', 'stop', 'step']
                for key in keys:
                    if key in selector and selector[key] is not None and not isinstance(selector[key], int):
                        raise RuntimeError('Slice selector for axis {0} must have ''{1}'' of type int.'.format(axis, key))


    # convert dict selectors into slice objects,
    # index and bool ndarrays
    def _parse_select(self, select_element):
        select = {axis: slice(None,None,None) for axis in self.df.dims.keys()}
        for axis, selector in iteritems(select_element):
            if isinstance(selector, list):
                if len(selector) == 0:
                    select[axis] = np.array([], dtype=np.int)
                elif all(type(x) == bool for x in selector):
                    select[axis] = np.array(selector, dtype=np.bool)
                elif all(isinstance(x, int) for x in selector):
                    select[axis] = np.array(selector, dtype=np.int)
                else:
                    raise RuntimeError('All elements of selector for axis {0} do not have the same type.'.format(axis))
            elif isinstance(selector, dict):
                select[axis] = slice(selector.get('start'), selector.get('stop'), selector.get('step'))
            elif isinstance(selector, int):
                select[axis] = selector
            elif selector is None:
                pass
            else:
                raise ValueError('Unexpected selector of type ' + str(type(selector)))
        return select


    # parse and convert all request selectors into index arrays
    def _get_subscripts_from_select(self, select_element):
        subscripts = self._parse_select(select_element)
        for axis, subs in iteritems(subscripts):
            if isinstance(subs, np.ndarray) and subs.dtype == np.bool:
                subscripts[axis] = subs.nonzero()[0]
            # slices are faster with xarray/dask; preserve them instead of converting to indices
            #elif isinstance(subs, slice):
            #    subscripts[axis] = np.array(range(*subs.indices(self.df.dims[axis])), dtype=np.int)
        #subscripts = np.ix_(*subscripts)
        return subscripts


    def check_fields_in_variables(self, fields, res=None):
        ''' Asks if a field or fields are in a dataset's variables.

        Parameters
        ----------
        fields : list or str
            Check these fields.
        res : xarray.Dataset, optional
            Check against the variables of this dataset. Defaults to self.df

        Raises
        ------
        ValueError : 
            If any fields are missing they will be reported here.

        '''

        if not fields:
            return

        if res is None:
            res = self.df

        if not isinstance(fields, list):
            fields = [fields]

        missing = []
        variables = list(res.variables)

        for field in fields:
            if not field in res:
                missing.append(field)

        if len(missing) > 0:
            raise ValueError('requested fields: {}, which are unavailable.'.format(', '.join(missing)))


    def _get_data(self, select=None, coords=None, fields=None, filters=None, dim_order=None, df=None, drop=False):
        if df is not None:
            res = df
        else:
            res = self.df
        
        self.check_fields_in_variables(fields, res)

        if filters:
            res, f = self._query(filters, df=res)
        subscripts=None
        if fields:
            res = res[fields]
        if select:
            self._validate_select(select)
            subscripts = self._get_subscripts_from_select(select)
        if subscripts:
            res = res.isel(**subscripts, drop=drop)
        if coords:
            # cast coords to correct type
            coords = {dim: np.array(v, dtype=res.coords[dim].dtype).tolist() for dim,v in iteritems(coords)}
            # apply all coords, rounding to nearest when possible
            for dim, coord in iteritems(coords):
                try:
                    res = res.sel(**{dim: coord}, drop=drop, method='nearest')
                except ValueError:
                    res = res.sel(**{dim: coord}, drop=drop)
        if filters and f['masks']:
            #ds = res.load()
            ds = res
            res = xr.Dataset()
            mask = reduce(xr_ufuncs.logical_and, f['masks'])
            if mask.nbytes <= self.max_cacheable_bytes:
                mask = mask.compute()
            for field in ds.variables:
                reduce_dims = [dim for dim in mask.dims if dim not in ds[field].dims]
                reduced = ds[field].where(mask.any(dim=reduce_dims), drop=True)
                if field in ds.data_vars:
                    res[field] = reduced
                else:
                    if field not in res.coords:
                        res.coords[field] = reduced.coords[field]
                for coord in res[field].coords:
                    res.coords[coord] = res[field].coords[coord]
        if dim_order:
            res = res.transpose(*dim_order)
        return res


    def _get_field(self, field, select=None, coords=None, filters=None, df=None, in_memory_only=False):
        if df is None:
            df = self.df
        data = self._get_data(select=select, coords=coords, df=df)
        data, f = self._query(filters, df=data)
        data = self._get_data(fields=field, df=data)
        if f['masks']:
            for mask in f['masks']:
                mask.attrs['is_mask'] = True
            mask_ds = xr.Dataset({'__mask_{}'.format(i): m for i,m in enumerate(f['masks'])})
        else:
            mask_ds = xr.Dataset({'__mask': xr.DataArray(np.array(True, dtype=np.bool), attrs={'is_mask': True})})
        ds = xr.Dataset({'data': data})
        ds = ds.merge(mask_ds, inplace=True)
        if in_memory_only:
            for field in ds.variables:
                if ds[field].nbytes > self.max_cacheable_bytes and not isinstance(ds[field].data, np.ndarray):
                    raise ValueError('cannot get large data field or mask that is not already loaded in-memory as an ndarray')
        return ds, f


    def info(self):
        dims = {name: size for name, size in self.df.dims.items()}
        variables = {name: {'type': da.dtype.name, 'dims': da.dims} for name, da in self.df.variables.items()}
        def serializable(x):
            try:
                json.dumps(x)
                return True
            except:
                return False

        for name, da in self.df.variables.items():
            variables[name]['attrs'] = {k: v for k, v in da.attrs.items() if serializable(v)}
        attrs = {k: v for k, v in self.df.attrs.items() if serializable(v)}
        return {'dims': dims, 'vars': variables, 'attrs': attrs}


    #todo: add sort and deprecate select()
    def raw(self, select=None, coords=None, fields=None, filters=None, dim_order=None):
        res = self._get_data(select, coords, fields, filters, dim_order)
        size = sum([res[field].nbytes for field in res.variables])
        if size > self.max_response_size:
            raise ValueError('Requested would return ' + str(size) + ' bytes; please limit request to ' + str(self.max_response_size) + '.')
        return res


    def image(self, select, coords, field, dim_order=None, image_format='jpeg'):
        if dim_order:
            if 'RGBA' in self.df[field].dims:
                if 'RGBA' in dim_order:
                    dim_order.remove('RGBA')
                dim_order.append('RGBA')
        data = self.raw(select=select, coords=coords, fields=[field], dim_order=dim_order)
        dims = list(data[field].dims)
        if 'RGBA' in dims:
            dims.remove('RGBA')
            dims.append('RGBA')
            data = data.transpose(*dims)
        data = data[field].values
        data = np.squeeze(data)

        if data.ndim != 2 and not (data.ndim == 3 and data.shape[2] == 4):
            raise RuntimeError('Non 2-d region selected when requesting image')

        #todo: probably ought to make normalization configurable in the request in some manner
        if data.ndim == 2 and data.dtype != np.uint8 and 'max' in self.stats and field in self.stats['max']:
            data = ((data / self.stats['max'][field]) * 255.0).astype(np.uint8)

        image = Image.fromarray(data)
        buf = BytesIO()
        if image_format.lower() == 'jpeg':
            image = image.convert('RGB')
            image.save(buf, format='JPEG', quality=40)
        elif image_format.lower() == 'png':
            image.save(buf, format='PNG')
        else:
            raise RuntimeError('Invalid or unsupported image format')
        #return {'data': bytes(buf.getvalue())}
        return {'data': 'data:image/' + image_format.lower() + ';base64,' + base64.b64encode(buf.getvalue()).decode()}


    def select(self,
               dim=None,
               filters=None,
               sort=None,
               ascending=None,
               start=0,
               stop=None,
               indexes=None,
               fields=None,
               options={}):
        r = self.df
        if dim is None:
            dim = list(r.dims)[0]
        redis_client = self.redis_client
        if not filters and not fields == "indexes_only":
            if indexes:
                num_results = np.array(indexes)[start:stop].size
            else:
                num_results = r[dim][start:stop].size
            if options.get('max_records') and num_results > options.get('max_records'):
                raise ValueError('Requested would return ' + str(num_results) + ' records; please limit request to ' + str(options['max_records']) + ' records.')
        inds = np.array(range(r.dims[dim]), dtype=np.int)
        if filters is not None:
            _, mask = self._query(filters)
            if 'dim_0' in mask['inds']:
                inds_elem = mask['inds']['dim_0']
                if isinstance(inds_elem, slice):
                    inds = inds[inds_elem]
                else:
                    inds = np.array(inds_elem)
            if mask['masks']:
                m = reduce(np.logical_and, mask['masks'])
                inds = inds[m]
        if indexes is not None:
            indexes = [x for x in indexes if x != None]
            if inds is not None:
                inds = inds[np.in1d(inds, indexes)]
            else:
                inds = np.array(indexes, dtype=np.int)
        if sort and r[dim].size > 0:
            sorted_inds = self._sort(dim, sort, ascending)
            if inds is not None:
                inds = sorted_inds[np.in1d(sorted_inds, inds)]
            else:
                inds = sorted_inds
        filtered_total = inds.size
        inds = inds[start:stop]
        if fields == "indexes_only":
            return inds, filtered_total
        else:
            if options.get('max_records') and inds.size > options.get('max_records'):
                raise ValueError('Requested would return ' + str(inds.size) + ' records; please limit request to ' + str(options['max_records']) + ' records.')
            if fields and type(fields) is list:
                for field in fields:
                    if field not in r.variables:
                        raise ValueError('Requested field \'' + str(field) + '\' does not exist. Allowable fields are: ' + str(r.variables))
                res = r[{dim: inds}][fields]
            else:
                res = r[{dim: inds}]
            return res


    def corr(self, field, dim, seed_idx, select=None, coords=None, filters=None):
        ds, f = self._get_field(field, select, coords, filters, in_memory_only=True)
        key_prefix = [self.name, 'mdata', field, select, filters]
        memoize = lambda key, f, *args, **kwargs: self._memoize(key_prefix+key, f, *args, **kwargs)
        res = par_correlation(ds, seed_idx, dim, self.num_chunks, self.max_workers, memoize=memoize)
        res = res.dropna(dim, how='all', subset=['corr'])
        return res


    def _memoize(self, key, f, *args, **kwargs):
        cached = self.redis_client.get(key)
        if not cached:
            res = f(*args, **kwargs)
            self.redis_client.setnx(key, pickle.dumps(res))
            return res
        else:
            return pickle.loads(cached)


    #todo: use _cache()
    def _sort(self, dim, sort, ascending):
        df = self.df
        sort_cache_key = json.dumps([self.name, 'sort', dim, sort, ascending])
        cached = self.redis_client.get(sort_cache_key)
        if not cached:
            if not ascending:
                ascending = [True] * len(sort)
            for field in sort:
                if field not in df.variables:
                    raise ValueError('Requested sort field \'' + str(field) + '\' does not exist. Allowable fields are: ' + str(df.variables))
            s = df[sort]
            ranks = np.zeros((s.dims[dim], len(s.variables)), dtype=np.int)
            for idx, (field, asc) in enumerate(zip(sort[::-1], ascending[::-1])):
                if s[field].dtype.name.startswith('string') or s[field].dtype.name.startswith('unicode') or s[field].dtype.name.startswith('bytes'):
                    blank_count = np.count_nonzero(s[field] == '')
                else:
                    blank_count = np.count_nonzero(np.isnan(s[field]))
                ranks[:,idx] = rankdata(s[field], method='dense')
                maxrank = np.max(ranks[:,idx])
                if not asc:
                    ranks[:,idx] = maxrank - (ranks[:,idx] - 1) - blank_count
                    ranks[:,idx][ranks[:,idx]<=0] = maxrank - blank_count + 1
                else:
                    ranks[:,idx] = np.clip(ranks[:,idx], 0, maxrank - blank_count + 1)
            res = np.lexsort(tuple(ranks[:,idx] for idx in range(ranks.shape[1])))
            self.redis_client.setnx(sort_cache_key, pickle.dumps(res))
            return res
        else:
            return pickle.loads(cached)


    def distance_filter(self, fields, point, value, df):

        operands = []
        for operand, coord in zip(fields, point):

            if not isinstance(operand, dict):
                operand = {'field': operand}

            field = operand['field']
            select = operand.get('select', None)
            coords = operand.get('coords', None)

            data = self._get_data(fields=field, select=select, coords=coords, df=df, drop=True)

            if df[field] is not self.df[field] or field not in self.argsorts or select or coords:
                data = data.where(xr_ufuncs.logical_and(data>=(coord-value), data<=(coord+value)), drop=True)

            else:
                min_point_ind = np.searchsorted(df[field].values.flat, coord-value, side='left', sorter=self.argsorts[field])
                max_point_ind = np.searchsorted(df[field].values.flat, coord+value, side='right', sorter=self.argsorts[field])
                min_point = np.unravel_index(self.argsorts[field][min_point_ind], df[field].shape)
                max_point = np.unravel_index(self.argsorts[field][max_point_ind], df[field].shape)
                data = df[field][{dim: slice(min_point[i], max_point[i]) for i,dim in enumerate(df[field].dims)}]

            operands.append(data-coord)

        mask = (reduce(operator.add, map(xr_ufuncs.square, operands)) ** 0.5) <= value
        return {'inds': {}, 'masks': [mask]}


    def isnan_filter(self, filt, dataset):
        ''' Masks values equivalent to nan

        Parameters
        ----------
        filt : dict
            filter clause
        dataset : xarray.DataSet
            Dataset to filter

        Returns
        -------
        response : dict 
            Keys are inds and masks. 

        '''

        field = filt['field']
        select = filt.get('select', None)
        coords = filt.get('coords', None)

        response = {'inds': {}, 'masks': []}

        if dataset is not self.df or field not in self.argsorts or select or coords:
            data = self._get_data(fields=field, select=select, coords=coords, df=dataset, drop=True)
            response['masks'].append(data.isnull())

        else:

            filter_key = json.dumps([self.name, 'filter', filt['op'], field, select, coords])
            cached = self.redis_client.get(filter_key)

            if not cached:
                start = np.searchsorted(dataset[field].values.flat, np.nan, side='left', sorter=self.argsorts[field])
                inds = np.sort(self.argsorts[field][start:])
                self.redis_client.setnx(filter_key, pickle.dumps(inds))
            else:
                inds = pickle.loads(cached)

            self.mask_from_filter_inds(dataset, field, inds, response=response)

        return response


    def mask_from_filter_inds(self, df, field, flat_inds, response=None):

        if response is None:
            response = {'inds': {}, 'masks': []}

        if 1 == df[field].ndim:
            key = df[field].dims[0]
            response['inds'][key] = xr.DataArray(flat_inds, dims=df[field].dims)
            return response

        unravel_inds = np.unravel_index(flat_inds, df[field].shape)
        for i, dim in enumerate(df[field].dims):
            response['inds'][dim] = xr.DataArray(np.unique(unravel_inds[i]), dims=dim)

        #todo: upgrade xarray and use this:
        #mask = xr.zeros_like(df[field], dtype=np.bool)

        mask = xr.DataArray(np.zeros_like(df[field].values, dtype=np.bool), dims=df[field].dims)
        mask.values.flat[flat_inds] = True
        response['masks'].append(mask)

        return response


    def _filter(self, f, df):

        if f['op'] == 'distance':
            return self.distance_filter(f['fields'], f['point'], f['value'], df)

        if f['op'] == 'isnan':
            return self.isnan_filter(f, df)

        op = f['op']
        field = f['field']
        value = f['value']
        select = f.get('select', None)
        coords = f.get('coords', None)

        def _filter_key(f, select, coords):
            return json.dumps([self.name, 'filter', f['op'], f['field'], f['value'], select, coords])

        res = {'inds': {}, 'masks': []}
        if df[field].dtype == 'bool' and op in ['=', 'is'] and value == True:
            res['masks'].append(df[field])
        elif df is not self.df or field not in self.argsorts or select or coords:
            lhs = self._get_data(fields=field, select=select, coords=coords, df=df, drop=True)
            if op == '=' or op == 'is':
                mask = (lhs == value)
            elif op == '<':
                mask = (lhs < value)
            elif op == '>':
                mask = (lhs > value)
            elif op == '<=':
                mask = (lhs <= value)
            elif op == '>=':
                mask = (lhs >= value)
            elif op == '!=':
                mask = (lhs != value)
            #todo: if df is self.df, could cache depending on size of mask
            #todo: convert 1-d masks to inds (?)
            res['masks'].append(mask)
        else:
            filter_key = _filter_key(f, select, coords)
            cached = self.redis_client.get(filter_key)
            if not cached:
                start = 0
                stop = df[field].size
                if op == '=' or op == 'is':
                    start = np.searchsorted(df[field].values.flat, value, side='left', sorter=self.argsorts[field])
                    stop = np.searchsorted(df[field].values.flat, value, side='right', sorter=self.argsorts[field])
                elif op == '<':
                    stop = np.searchsorted(df[field].values.flat, value, side='left', sorter=self.argsorts[field])
                elif op == '>':
                    start = np.searchsorted(df[field].values.flat, value, side='right', sorter=self.argsorts[field])
                elif op == '<=':
                    stop = np.searchsorted(df[field].values.flat, value, side='right', sorter=self.argsorts[field])
                elif op == '>=':
                    start = np.searchsorted(df[field].values.flat, value, side='left', sorter=self.argsorts[field])
                inds = np.sort(self.argsorts[field][start:stop])
                self.redis_client.setnx(filter_key, pickle.dumps(inds))
            else:
                inds = pickle.loads(cached)
            
            self.mask_from_filter_inds(df, field, inds, response=res)

        return res


    def _query(self, filters, df=None):
        if df is None:
            df = self.df
        if isinstance(filters, list):
            filters = {'and': filters}
        if not filters:
            return (df, {'inds': {}, 'masks': []})
        else:
            #todo: add ability to filter on a boolean field without any op/value
            #todo: add not-equals

            def _and(m1, m2):
                res = {'inds': {}, 'masks': []}
                d = {}
                for mask in m1['masks']+m2['masks']:
                    d.setdefault(mask.dims, []).append(mask)
                for k, v in iteritems(d):
                    res.setdefault('masks', []).append(reduce(xr_ufuncs.logical_and, v))
                res['inds'] = m1['inds']
                for dim, inds in iteritems(m2['inds']):
                    if dim in res['inds']:
                        res['inds'][dim] = np.intersect1d(res['inds'][dim], inds, assume_unique=True)
                    else:
                        res['inds'][dim] = inds
                return res

            def _or(m1, m2):
                res = {'inds': {}, 'masks': []}

                for dim, inds in iteritems(m1['inds']):
                    mask = xr.DataArray(np.zeros((df.dims[dim],), dtype=np.bool), dims=[dim])
                    mask.coords[dim] = df.coords[dim]
                    mask[inds] = True
                    m1['masks'].append(mask)
                for dim, inds in iteritems(m2['inds']):
                    mask = xr.DataArray(np.zeros((df.dims[dim],), dtype=np.bool), dims=[dim])
                    mask.coords[dim] = df.coords[dim]
                    mask[inds] = True
                    m2['masks'].append(mask)

                if not m1['masks']:
                    if not m2['masks']:
                        res['masks'] = []
                    else:
                        res['masks'] = [reduce(xr_ufuncs.logical_and, m2['masks'])]
                elif not m2['masks']:
                    res['masks'] = [reduce(xr_ufuncs.logical_and, m1['masks'])]
                else:
                    res['masks'] = [xr_ufuncs.logical_or(reduce(xr_ufuncs.logical_and, m1['masks']), reduce(xr_ufuncs.logical_and, m2['masks']))]
                return res

            def _expand_filters(filters):
                if isinstance(filters, dict):
                    for op in ['and', 'or', 'any', 'all', 'count']:
                        if op in filters:
                            ret = filters
                            ret[op] = _expand_filters(ret[op])
                            return ret

                    op = filters['op']
                    if op == 'between':
                        and_clause = {'and': []}
                        and_clause['and'].append({'op': '>=', 'field': filters['field'], 'value': filters['value'][0]})
                        and_clause['and'].append({'op': '<=', 'field': filters['field'], 'value': filters['value'][1]})
                        return and_clause
                    elif op == 'in':
                        or_clause = {'or': []}
                        for v in filters['value']:
                            or_clause['or'].append({'op': '=', 'field': filters['field'], 'value': v})
                        return or_clause
                    else:
                        return filters
                    return res
                elif isinstance(filters, list):
                    return list(map(_expand_filters, filters))
                else:
                    assert False
            
            filters = _expand_filters(filters)

            def _reduce(filters):
                if isinstance(filters, dict):
                    if 'inds' in filters or 'masks' in filters:
                        return filters
                    elif 'and' in filters or 'or' in filters:
                        if 'and' in filters:
                            filters = filters['and']
                            bool_func = _and
                        elif 'or' in filters:
                            filters = filters['or']
                            bool_func = _or
                        else:
                            assert False
                        return reduce(bool_func, _reduce(filters), {'inds': {}, 'masks': []})
                    elif 'any' in filters or 'all' in filters or 'count' in filters:
                        dims = filters['dims']
                        if not isinstance(dims, list):
                            dims = [dims]
                        if 'any' in filters:
                            filters = filters['any']
                            def _any(m):
                                for dim, inds in iteritems(m['inds']):
                                    mask = xr.DataArray(np.zeros((df.dims[dim],), dtype=np.bool), dims=[dim])
                                    mask.coords[dim] = df.coords[dim]
                                    mask[inds] = True
                                    m['masks'].append(mask)
                                mask = reduce(xr_ufuncs.logical_and, m['masks'])
                                if not set(dims).isdisjoint(mask.dims):
                                    mask = mask.any(dim=set(dims).intersection(mask.dims))
                                return {'inds': {}, 'masks': [mask]}
                            agg_func = _any
                        elif 'all' in filters:
                            filters = filters['all']
                            def _all(m):
                                res = {'inds': {dim: m['inds'][dim] for dim in m['inds'] if dim not in dims}, 'masks': []}
                                for mask in m['masks']:
                                    if not set(dims).isdisjoint(mask.dims):
                                        mask = mask.all(dim=set(dims).intersection(mask.dims))
                                    res['masks'].append(mask)
                                return res
                            agg_func = _all
                        elif 'count' in filters:
                            op = filters['op']
                            value = filters['value']
                            filters = filters['count']
                            def _count(m):
                                res = {'inds': {dim: m['inds'][dim] for dim in m['inds'] if dim not in dims}, 'masks': []}
                                for mask in m['masks']:
                                    if not set(dims).isdisjoint(mask.dims):
                                        count = mask.sum(dim=set(dims).intersection(mask.dims))
                                        if op == '=' or op == 'is':
                                            mask = (count == value)
                                        elif op == '<':
                                            mask = (count < value)
                                        elif op == '>':
                                            mask = (count > value)
                                        elif op == '<=':
                                            mask = (count <= value)
                                        elif op == '>=':
                                            mask = (count >= value)
                                        elif op == '!=':
                                            mask = (count != value)
                                    res['masks'].append(mask)
                                return res
                            agg_func = _count
                        else:
                            assert False
                        return agg_func(_reduce(filters))
                    else:
                        return self._filter(filters, df)
                elif isinstance(filters, list):
                    return list(map(_reduce, filters))
                else:
                    return {'inds': {}, 'masks': []}


            res = _reduce(filters)

            for dim in df.dims.keys():
                if dim in res['inds'] and res['inds'][dim].size > 0:
                    inds_min = int(res['inds'][dim].min())
                    inds_max = int(res['inds'][dim].max())+1
                    if inds_max-inds_min == res['inds'][dim].size:
                        if inds_min == 0:
                            inds_min = None
                        if inds_max == df.dims[dim]:
                            inds_max = None
                        if inds_min or inds_max:
                            res['inds'][dim] = slice(inds_min,inds_max)
                        else:
                            res['inds'].pop(dim, None)

            res['masks'] = [mask[{dim: res['inds'][dim] for dim in set(res['inds'].keys()).intersection(mask.dims)}] for mask in res['masks']]

            return (df[res['inds']], res)
