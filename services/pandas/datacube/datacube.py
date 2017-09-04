import numpy as np
from scipy.stats import rankdata
import json
import pickle
import redis
#import txredisapi
#from twisted.internet import reactor
import xarray as xr
import xarray.ufuncs as xr_ufuncs
import dask
import dask.array as da
from PIL import Image
from io import BytesIO
import base64
from functools import reduce

from six import iteritems
from builtins import int, map


#todo: https://github.com/dask/dask/issues/732
def da_einsum(*args, **kwargs):
    args = list(args)
    out_inds = args.pop()
    in_inds = args[1::2]
    arrays = args[0::2]
    dtype = kwargs.get('dtype')
    einsum_dtype = dtype
    if dtype is None:
        dtype = np.result_type(*[a.dtype for a in arrays])
    casting = kwargs.get('casting')
    if casting is None:
        casting = 'safe'
    
    full_inds = list(set().union(*in_inds))
    contract_inds = tuple(set(full_inds)-set(out_inds))
    
    def einsum(*operands, **kwargs):
        kernel_dtype = kwargs.get('kernel_dtype')
        casting = kwargs.get('casting')
        chunk = np.einsum(dtype=kernel_dtype, casting=casting, *([arg for arg_pair in zip(operands, in_inds) for arg in arg_pair]+[out_inds]))
        chunk = np.array(chunk)
        chunk.shape = tuple([1 if ind not in out_inds else chunk.shape[out_inds.index(ind)] for ind in full_inds])
        return chunk
    
    adjust_chunks = {ind: 1 for ind in contract_inds}
    result = da.atop(einsum, full_inds, *args, dtype=dtype, kernel_dtype=einsum_dtype, casting=casting, adjust_chunks=adjust_chunks)
    if contract_inds:
        result = da.sum(result, axis=contract_inds)
    return result

# monkey-patch the method into the module for now
setattr(da, 'einsum', da_einsum)


class Datacube:


    def test(self):
        return 1


    def __init__(self, name, nc_file, redis_client=None, chunks=None, max_response_size=10*1024*1024, max_cacheable_bytes=10*1024*1024):
        self.name = name
        self.max_response_size = max_response_size
        self.max_cacheable_bytes = max_cacheable_bytes
        if nc_file: self.load(nc_file, chunks)
        #todo: would be nice to find a way to swap these out,
        # and also to be able to run without redis (numpy-only)
        #if reactor.running:
        #    self.redis_client = txredisapi.Connection('localhost', 6379)
        #else:
        if redis_client:
            self.redis_client = redis_client
        else:
            self.redis_client = redis.StrictRedis('localhost', 6379)


    def load(self, nc_file, chunks=None):
        #todo: rename df
        #todo: argsorts need to be cached to a file (?)
        self.df = xr.open_dataset(nc_file, chunks=chunks)
        #todo: rework _query so this is not needed:
        for dim in self.df.dims:
            if dim not in self.df.keys():
                self.df.coords[dim] = range(self.df.dims[dim])
        #todo: add option whether to create mdata
        #self.mdata = xr.ufuncs.logical_not(xr.ufuncs.isnan(self.df))
        #self.mdata = self.mdata.persist()
        #todo: can nan-only chunks be dropped?
        self.df = self.df.fillna(0.)
        self.df = self.df.persist()
        if chunks:
            self.backend = da
        else:
            self.backend = np
        self.argsorts = {}
        for field in self.df.keys():
            if self.df[field].size*np.dtype('int64').itemsize <= self.max_cacheable_bytes:
                self.argsorts[field] = np.argsort(self.df[field].values, axis=None)
        #todo: would be nice to cache these instead of computing on every startup
        self.mins = {}
        self.maxes = {}
        self.means = {}
        self.stds = {}
        for field in self.df.keys():
            self.mins[field] = self.df[field].min().values
            self.maxes[field] = self.df[field].max().values
            if not self.df[field].dtype.kind in 'OSU':
                self.means[field] = self.df[field].mean().values
                self.stds[field] = self.df[field].std().values


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


    def _get_data(self, select=None, fields=None, filters=None, dim_order=None, df=None):
        if df is not None:
            res = df
        else:
            res = self.df
        if isinstance(fields, list):
            assert(all(f in list(res.keys()) for f in fields))
        elif fields:
            assert(fields in list(res.keys()))
        subscripts=None
        if select:
            self._validate_select(select)
            subscripts = self._get_subscripts_from_select(select)
        if subscripts:
            res = res[subscripts]
        if filters:
            res, f = self._query(filters, df=res)
            if f['masks']:
                ds = res.load()
                res = xr.Dataset()
                mask = reduce(xr_ufuncs.logical_and, f['masks'])
                for field in ds.data_vars:
                    reduce_dims = [dim for dim in mask.dims if dim not in ds[field].dims]
                    res[field] = ds[field].where(mask.any(dim=reduce_dims), drop=True)
                    for coord in res[field].coords:
                        res.coords[coord] = res[field].coords[coord]
        if fields:
            res = res[fields]
        if dim_order:
            res = res.transpose(*dim_order)
        return res


    def info(self):
        dims = {name: size for name, size in self.df.dims.items()}
        variables = {name: {'type': da.dtype.name, 'dims': da.dims} for name, da in self.df.variables.items()}
        for name, da in self.df.variables.items():
            variables[name]['attrs'] = dict(da.attrs.items())
        attrs = dict(self.df.attrs.items())
        return {'dims': dims, 'vars': variables, 'attrs': attrs}


    #todo: add sort and deprecate select()
    def raw(self, select=None, fields=None, filters=None, dim_order=None):
        res = self._get_data(select, fields, filters, dim_order)
        size = sum([res[field].size*res[field].dtype.itemsize for field in res.keys()])
        if size > self.max_response_size:
            raise ValueError('Requested would return ' + str(size) + ' bytes; please limit request to ' + str(self.max_response_size) + '.')
        return res


    def image(self, select, field, dim_order=None, image_format='jpeg'):
        if dim_order:
            if 'RGBA' in self.df[field].dims:
                if 'RGBA' in dim_order:
                    dim_order.remove('RGBA')
                dim_order.append('RGBA')
        data = self.raw(select=select, fields=[field], dim_order=dim_order)
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
        if data.ndim == 2 and data.dtype != np.uint8:
            data = ((data / self.maxes[field]) * 255.0).astype(np.uint8)

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
                inds = np.array(mask['inds']['dim_0'])
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
                    if field not in r.keys():
                        raise ValueError('Requested field \'' + str(field) + '\' does not exist. Allowable fields are: ' + str(r.keys()))
                res = r[{dim: inds}][fields]
            else:
                res = r[{dim: inds}]
            return res


    def corr(self, field, dim, seed_idx, select=None, filters=None):
        data = self._get_data(select=select)
        data, f = self._query(filters, df=data)
        #todo: use self.mdata if it exists
        #mdata = self.mdata[field].reindex_like(data)
        data = self._get_data(fields=field, df=data)
        mdata = xr.DataArray(da.from_array(np.ones((1,)*data.ndim, dtype=np.bool), chunks=1), dims=['mdim_{0}'.format(i) for i in range(data.ndim)])
        if seed_idx not in data.coords[dim].values:
            res = xr.Dataset({'corr': ([dim], np.full((data.coords[dim].size,), np.nan))})
        else:
            #todo: more xarray-esque way of doing the following:
            seed_idx = np.where(data.coords[dim].values==seed_idx)[0][0]
            if f['masks']:
                if len(f['masks'])>1:
                    mdata = reduce(xr_ufuncs.logical_and, f['masks'], mdata)
                else:
                    mdata = f['masks'][0]
                mdata = mdata.any(dim=set(mdata.dims)-set(data.dims))
                if data.chunks is not None:
                    mdata = mdata.chunk(tuple(data.chunks[data.dims.index(dim)] for dim in mdata.dims))
                mdata = mdata.expand_dims(set(data.dims)-set(mdata.dims))
                mdata = mdata.transpose(*data.dims)
            axis = data.dims.index(dim)
            if mdata.size<data.size:
                mdata = mdata.astype(data.dtype)
            if dim in mdata.dims and mdata.sizes[dim]==1:
                bounds = tuple(slice(inds.min(), inds.max()+1) if mdata.shape[i]>1 and i!=axis else slice(None) for i,inds in enumerate(np.nonzero(mdata.values)))
                data = data[bounds]
                mdata = mdata[bounds]
            res = Datacube._corr(data.data, mdata.data, seed_idx, axis)
            res = xr.Dataset({'corr': ([dim], res.squeeze()), dim: data.coords[dim]})
        if f['masks']:
            mask = reduce(xr_ufuncs.logical_and, f['masks'])
            reduce_dims = [d for d in mask.dims if d != dim]
            res = res.reindex_like(mask).where(mask.any(dim=reduce_dims), drop=True)
        return res


    @staticmethod
    def _get_num_samples(data, axis, mdata, backend=np):
        sample_axes = np.array([i for i in range(data.ndim) if i != axis])
        if mdata is None:
            num_samples = np.prod(np.array([data.shape[i] for i in sample_axes], dtype=np.uint64))
            if backend == da:
                num_samples = da.from_array(num_samples, chunks=data.chunks)
            mdata_args = []
        else:
            dims = range(mdata.ndim)
            #num_samples = backend.einsum(mdata.astype(np.uint64), dims, [axis], dtype=np.uint64)
            num_samples = backend.count_nonzero(mdata, axis=tuple(sample_axes))
            num_samples = backend.reshape(num_samples, tuple(mdata.shape[i] if i == axis else 1 for i in range(mdata.ndim)))
            mdata_args = [mdata, dims]
            num_samples *= np.prod(np.array([data.shape[i] for i in sample_axes if mdata.shape[i]==1 and data.shape[i]>1], dtype=np.uint64))
        return num_samples, mdata_args


    @staticmethod
    def _get_seed(data, seed_idx, axis, mdata, backend=np):
        seed = backend.take(data, seed_idx, axis=axis)
        seed = backend.reshape(seed, tuple(data.shape[i] if i != axis else 1 for i in range(data.ndim)))
        mseed = backend.take(mdata, seed_idx, axis=axis)
        mseed = backend.reshape(mseed, tuple(mdata.shape[i] if i != axis else 1 for i in range(data.ndim)))
        return seed, mseed


    #todo: use xr.DataArray's as parameters to this function instead of just dask arrays
    @staticmethod
    def _corr(data, mdata, seed_idx, axis):
        np.seterr(divide='ignore', invalid='ignore')

        def _compute(arr):
            if isinstance(arr, da.Array):
                tmp = arr.compute()
                return da.from_array(tmp, chunks=arr.chunks)
            else:
                return arr

        if isinstance(data, da.Array):
            backend = da
        else:
            backend = np

        sample_axes = np.array([i for i in range(data.ndim) if i != axis])
        seed, mseed = Datacube._get_seed(data, seed_idx, axis, mdata, backend=backend)

        mseed2 = mseed * ~np.isnan(seed)
        num_samples, _ = Datacube._get_num_samples(data, axis, mdata, backend=backend)
        seed_num_samples, _ = Datacube._get_num_samples(seed, axis, mseed2, backend=backend)
        seed_mean = backend.einsum(seed, range(seed.ndim), mseed2, range(mseed2.ndim), [], dtype=data.dtype) / seed_num_samples.astype(data.dtype)
        seed_dev = backend.einsum(seed-seed_mean, range(seed.ndim), mseed2, range(mseed2.ndim), range(seed.ndim), dtype=data.dtype)
        seed_denominator = backend.sum(seed_dev**2, axis=tuple(sample_axes))

        #data_mean = np.nanmean(data*backend.sign(np.inf*mdata), axis=tuple(sample_axes), keepdims=True)

        #data_mean = backend.einsum(data, range(data.ndim), mdata, range(mdata.ndim), [axis]) / num_samples
        #data_mean = backend.reshape(data_mean, tuple(data.shape[i] if i == axis else 1 for i in range(data.ndim)))

        def _sum(data, mdata, axis):
            sample_axes = np.array([i for i in range(data.ndim) if i != axis])
            result_shape = tuple(data.shape[i] if i==axis else 1 for i in range(data.ndim))
            if mdata.size<data.size and np.all(0==mdata):
                return np.zeros(result_shape, dtype=data.dtype)
            else:
                #return np.sum(data*mdata, axis=tuple(sample_axes), keepdims=True)
                res = np.einsum(data, range(data.ndim), mdata, range(data.ndim), [axis], dtype=data.dtype)
                res = np.reshape(res, result_shape)
                return res

        if isinstance(data, da.Array):
            data_mean = da.map_blocks(_sum, data, mdata, axis, dtype=data.dtype)
            data_mean = np.sum(data_mean, axis=tuple(sample_axes), keepdims=True)
            data_mean = _compute(data_mean)
        else:
            data_mean = _sum(data, mdata, axis)
        data_sum = data_mean
        data_mean = data_mean / num_samples.astype(data.dtype)

        def _einsum_corr_chunk(data, data_mean, seed, axis, mdata, mseed, seed_mean):
            if mdata.size<data.size and np.all(0==mdata):
                return np.zeros(data_mean.shape+(2,), dtype=data.dtype)
            else:
                np.seterr(divide='ignore', invalid='ignore')

                dtype = data.dtype
                sdims = range(seed.ndim)
                ddims = range(data.ndim)
                sample_axes = tuple(i for i in ddims if i != axis)

                seed_dev = np.einsum(seed-seed_mean, sdims, mseed, range(mseed.ndim), sdims, dtype=dtype)
                numerator = np.einsum(seed_dev, ddims, data, ddims, mdata, range(mdata.ndim), [axis], dtype=dtype)
                numerator -= np.einsum(seed_dev, ddims, data_mean, ddims, mdata, range(mdata.ndim), [axis], dtype=dtype)

                data_denominator = np.einsum(data, ddims, data, ddims, mdata, range(mdata.ndim), [axis], dtype=dtype)

                numerator = np.reshape(numerator, tuple(data.shape[i] if i == axis else 1 for i in range(data.ndim)))
                data_denominator = np.reshape(data_denominator, tuple(data.shape[i] if i == axis else 1 for i in range(data.ndim)))
                
                return np.concatenate([numerator[...,np.newaxis], data_denominator[...,np.newaxis]], axis=data.ndim)

        if isinstance(data, da.Array):
            result = da.map_blocks(_einsum_corr_chunk, data, data_mean, seed, axis, mdata, mseed, seed_mean, new_axis=data.ndim, dtype=data.dtype)
            result = result.compute()
        else:
            result = _einsum_corr_chunk(data, data_mean, seed, axis, mdata, mseed, seed_mean)
        numerator = result[..., 0]
        data_denominator = result[..., 1]
        numerator = np.sum(numerator, axis=tuple(sample_axes), keepdims=True)
        data_denominator = np.sum(data_denominator, axis=tuple(sample_axes), keepdims=True)
        data_denominator = data_denominator+data_mean**2*num_samples
        data_denominator = data_denominator-2.0*data_mean*data_sum

        denominator = np.sqrt(seed_denominator*data_denominator)
        corr = np.clip(numerator/denominator, -1.0, 1.0)
        both_samples, _ = Datacube._get_num_samples(data, axis, mdata*mseed, backend=backend)
        corr = corr * backend.sign(np.inf*(both_samples>1))
        corr = np.reshape(corr, tuple(data.shape[i] if i == axis else 1 for i in range(data.ndim)))
        return corr


    def _sort(self, dim, sort, ascending):
        df = self.df
        sort_cache_key = json.dumps([self.name, 'sort', dim, sort, ascending])
        cached = self.redis_client.get(sort_cache_key)
        if not cached:
            if not ascending:
                ascending = [True] * len(sort)
            for field in sort:
                if field not in df.keys():
                    raise ValueError('Requested sort field \'' + str(field) + '\' does not exist. Allowable fields are: ' + str(df.keys()))
            s = df[sort]
            ranks = np.zeros((s.dims[dim], len(s.keys())), dtype=np.int)
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


    #todo: 'select' clauses could be pretty easily added into each filter as in {'field': ..., 'select': ..., 'op': ..., 'value': ...}
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
            def _filter(f):
                op = f['op']
                field = f['field']
                value = f['value']

                def _filter_key(f):
                    return json.dumps([self.name, 'filter', f['op'], f['field'], f['value']])

                res = {'inds': {}, 'masks': []}
                if df is not self.df or field not in self.argsorts:
                    if op == '=' or op == 'is':
                        mask = (df[field] == value)
                    elif op == '<':
                        mask = (df[field] < value)
                    elif op == '>':
                        mask = (df[field] > value)
                    elif op == '<=':
                        mask = (df[field] <= value)
                    elif op == '>=':
                        mask = (df[field] >= value)
                    #todo: convert 1-d masks to inds (?)
                    res['masks'].append(mask)
                else:
                    filter_key = _filter_key(f)
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
                    
                    if 1==df[field].ndim:
                        res['inds'][df[field].dims[0]] = xr.DataArray(inds, dims=df[field].dims)
                    else:
                        unravel_inds = np.unravel_index(inds, df[field].shape)
                        for i, dim in enumerate(df[field].dims):
                            res['inds'][dim] = xr.DataArray(np.unique(unravel_inds[i]), dims=dim)
                        #todo: upgrade xarray and use this:
                        #mask = xr.zeros_like(df[field], dtype=np.bool)
                        mask = xr.DataArray(np.zeros_like(df[field].values, dtype=np.bool), dims=df[field].dims)
                        mask.values.flat[inds] = True
                        res['masks'].append(mask)
                return res

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
                    mask[inds] = True
                    m1['masks'].append(mask)
                for dim, inds in iteritems(m2['inds']):
                    mask = xr.DataArray(np.zeros((df.dims[dim],), dtype=np.bool), dims=[dim])
                    mask[inds] = True
                    m2['masks'].append(mask)

                if not m1['masks']:
                    if not m2['masks']:
                        res['masks'] = []
                    else:
                        res['masks'] = [reduce(xr_ufuncs.logical_and, m2['masks'])]
                elif not m2['masks']:
                    res['masks'] = []
                else:
                    res['masks'] = [xr_ufuncs.logical_or(reduce(xr_ufuncs.logical_and, m1['masks']), reduce(xr_ufuncs.logical_and, m2['masks']))]
                return res

            def _expand_filters(filters):
                bool_op = None
                if 'and' in filters:
                    bool_op = 'and'
                elif 'or' in filters:
                    bool_op = 'or'

                res = {bool_op: []}
                for f in filters[bool_op]:
                    if 'and' in f or 'or' in f:
                        res[bool_op].append(_expand_filters(f))
                    else:
                        op, field, value = (f['op'], f['field'], f['value'])
                        if op == 'between':
                            and_clause = {'and': []}
                            and_clause['and'].append({'op': '>=', 'field': field, 'value': value[0]})
                            and_clause['and'].append({'op': '<=', 'field': field, 'value': value[1]})
                            res[bool_op].append(and_clause)
                        elif op == 'in':
                            or_clause = {'or': []}
                            for v in value:
                                or_clause['or'].append({'op': '=', 'field': field, 'value': v})
                            res[bool_op].append(or_clause)
                        else:
                            res[bool_op].append(f)
                return res
            
            filters = _expand_filters(filters)

            def _reduce(filters):
                res = None
                bool_op = None
                bool_func = None
                if 'and' in filters:
                    bool_op = 'and'
                    bool_func = _and
                elif 'or' in filters:
                    bool_op = 'or'
                    bool_func = _or
                else:
                    if 'inds' in filters or 'masks' in filters:
                        return filters
                    else:
                        return _filter(filters)

                if not filters[bool_op]:
                    res = {'inds': {}, 'masks': []}
                elif any(isinstance(x, dict) and 'and' in x or 'or' in x for x in filters[bool_op]):
                    res = _reduce({bool_op: list(map(_reduce, filters[bool_op]))})
                else:
                    res = reduce(bool_func, list(map(_reduce, filters[bool_op])))
                return res

            res = _reduce(filters)
            res['masks'] = [mask[res['inds']] for mask in res['masks']]

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

            return (df[res['inds']], res)
