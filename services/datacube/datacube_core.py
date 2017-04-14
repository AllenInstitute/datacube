import numpy as np
import scipy as sp

import error

def _mask_args(axis, ndim, query):
    return lambda axes=range(0,ndim), out_axes=[axis], more=[]: \
        list(sum([(mask.squeeze(),[i]) for i, mask in enumerate(query) if i in axes and mask.size>1], ()))+more+[out_axes]

def correlation(data, seed, axis, query=None, mdata=None, mseed=None):
    """Compute correlation for each sample versus a seed sample, along an axis of a multi-dimensional array.

    Compute the correlation of data.take(i, axis=axis) and seed for all i.

    Memory usage beyond the input arguments themselves is negligible. Without costing additional memory, the
    computation can be restricted along any axis with a 1-D boolean array in `query` (consider using same
    dtype as `data`, however, for performance).

    Missing values can be specified in arrays `mdata` and `mseed` and will be omitted from the calculation.
    Any entry in the output where no samples are included will be set to np.nan.

    Args:
        data (numpy.ndarray): N-dimensional array with N >= 2, of any float dtype.
        seed (numpy.ndarray): Array with shape data.take(0, axis=axis).shape, and same dtype as `data`.
        axis (int): Axis to operate along, in range(0, data.ndim)
        query (Optional[list]): list of length data.ndim, where the ith element is either numpy.ones(1)
            ("all") or an array of shape=(data.shape[i],) containing 0's and 1's where the corresponding
            index will be included in the calculation iff 1. dtype is not restricted to bool despite
            containing boolean values; performance may be improved by using, for example, dtype=np.float32
            when `data` also has dtype np.float32.
        mdata (Optional[numpy.ndarray]): Array of same shape and dtype as `data` where 1 indicates an
            an observed entry and 0 is a missing entry.
        mseed (Optional[numpy.ndarray]): Array of same shape and dtype as `seed` with the same semantics
            as `mdata`.

    Returns:
        numpy.ndarray: Array of shape (data.shape[axis],) where the ith entry is the correlation of
            data.take(i, axis=axis) with seed.
    """

    import numpy as np # import within this function in case it is being pushed to a distributed node
    
    if query is None:
        query = [np.ones(1, dtype=data.dtype)]*data.ndim

    mask_args = _mask_args(axis, data.ndim, query)
    
    sdims = range(0,seed.ndim)
    ddims = range(0,data.ndim)
    sample_axes = np.array([i for i in ddims if i != axis])

    if mdata is None:
        seed_num_samples = np.prod(np.array([query[i].sum() if query[i].size>1 else data.shape[i] for i in sample_axes]))
        num_samples = seed_num_samples
        mseed_args = []
        mdata_args = []
    else:
        seed_num_samples = np.einsum(mseed, range(0,mseed.ndim), *mask_args(sample_axes, []), dtype=np.int)
        num_samples = np.einsum(mdata, range(0,mdata.ndim), *mask_args(), dtype=np.int)
        mseed_args = [mseed, range(0, mseed.ndim)]
        mdata_args = [mdata, range(0, mdata.ndim)]

    seed_mean = np.einsum(seed, sdims, *mask_args(sample_axes, [], mseed_args)) / seed_num_samples
    data_mean = np.einsum(data, ddims, *mask_args(more=mdata_args)) / num_samples
    data_mean *= query[axis].flat
    data_mean.shape = tuple(data.shape[i] if i == axis else 1 for i in ddims) # restore dims after einsum
    
    seed_dev = np.einsum(seed-seed_mean, sdims, *mask_args(sample_axes, sdims, mseed_args))
    numerator = np.einsum(seed_dev, ddims, data, ddims, query[axis].flat, [axis], *mask_args((), [axis], mdata_args))
    numerator -= np.einsum(seed_dev, ddims, data_mean, ddims, [axis])

    denominator = np.einsum(data, ddims, data, ddims, *mask_args(more=mdata_args))
    # data_mean has already been masked
    denominator += -2.0*np.einsum(data, ddims, data_mean, ddims, *mask_args(sample_axes, more=mdata_args))
    denominator += np.sum(data_mean**2, axis=tuple(sample_axes)) * num_samples
    denominator *= np.einsum(seed_dev**2, sdims, *mask_args(sample_axes))
    denominator = np.sqrt(denominator)

    return np.clip(numerator / denominator, -1.0, 1.0) / query[axis].flat # intentional divide by zero

def _local_differential(differential_function, darr, axis, domain1, domain2, marr, **kwargs):
    import numpy as np

    domain2[axis] = domain1[axis]
    if 'paired' in kwargs:
        for paired_axis in kwargs['paired']:
            domain2[paired_axis] = domain1[paired_axis]

    d1_query, d1_data, d1_mdata, d1_empty = _get_local_query(darr, marr, domain1)
    d2_query, d2_data, d2_mdata, d2_empty = _get_local_query(darr, marr, domain2)

    # skip computation and return sensible outputs when one of the domains is empty
    if axis in d1_empty:
        return np.array([], dtype=darr.ndarray.dtype)
    elif len(d1_empty)>0 or len(d2_empty)>0:
        result = np.empty((darr.local_shape[axis]), dtype=darr.ndarray.dtype)
        result.fill(np.nan)
        return result

    return globals()[differential_function](d1_data, d2_data, axis, d1_query, d2_query, d1_mdata, d2_mdata, **kwargs)

def fold_change(d1_data, d2_data, axis, domain1=None, domain2=None, d1_mdata=None, d2_mdata=None):
    import numpy as np
    
    assert(d1_data.ndim == d2_data.ndim)
    ndim = d1_data.ndim
    if domain1 is None:
        domain1 = [np.ones(1, dtype=d1_data.dtype)]*ndim
    if domain2 is None:
        domain2 = [np.ones(1, dtype=d2_data.dtype)]*ndim

    d1_mask_args = _mask_args(axis, ndim, domain1)
    d2_mask_args = _mask_args(axis, ndim, domain2)

    ddims = range(0,ndim)
    sample_axes = np.array([i for i in ddims if i != axis])

    if d1_mdata is None:
        d1_num_samples = np.prod(np.array([domain1[i].sum() if domain1[i].size>1 else d1_data.shape[i] for i in sample_axes]))
        d1_mdata_args = []
    else:
        d1_num_samples = np.einsum(d1_mdata, ddims, *d1_mask_args(), dtype=np.int)
        d1_mdata_args = [d1_mdata, range(0, ndim)]

    d1_mean = np.einsum(d1_data, ddims, *d1_mask_args(more=d1_mdata_args)) / d1_num_samples
    d1_mean *= domain1[axis].flat

    if d2_mdata is None:
        d2_num_samples = np.prod(np.array([domain2[i].sum() if domain2[i].size>1 else d2_data.shape[i] for i in sample_axes]))
        d2_mdata_args = []
    else:
        d2_num_samples = np.einsum(d2_mdata, ddims, *d2_mask_args(), dtype=np.int)
        d2_mdata_args = [d2_mdata, range(0, ndim)]
    
    d2_mean = np.einsum(d2_data, ddims, *d2_mask_args(more=d2_mdata_args)) / d2_num_samples
    d2_mean *= domain2[axis].flat

    return np.divide(d1_mean, d2_mean) / domain1[axis].flat # intentional divide by zero

def differential(d1_data, d2_data, axis, domain1=None, domain2=None, d1_mdata=None, d2_mdata=None):
    import numpy as np
    from scipy.stats import t

    assert(d1_data.ndim == d2_data.ndim)
    ndim = d1_data.ndim
    if domain1 is None:
        domain1 = [np.ones(1, dtype=d1_data.dtype)]*ndim
    if domain2 is None:
        domain2 = [np.ones(1, dtype=d2_data.dtype)]*ndim

    d1_mask_args = _mask_args(axis, ndim, domain1)
    d2_mask_args = _mask_args(axis, ndim, domain2)

    ddims = range(0,ndim)
    sample_axes = np.array([i for i in ddims if i != axis])

    if d1_mdata is None:
        d1_num_samples = np.prod(np.array([domain1[i].sum() if domain1[i].size>1 else d1_data.shape[i] for i in sample_axes]))
        d1_mdata_args = []
    else:
        d1_num_samples = np.einsum(d1_mdata, ddims, *d1_mask_args(), dtype=np.int)
        d1_mdata_args = [d1_mdata, range(0, ndim)]

    d1_mean = np.einsum(d1_data, ddims, *d1_mask_args(more=d1_mdata_args)) / d1_num_samples
    d1_mean *= domain1[axis].flat
    d1_var = np.einsum(d1_data, ddims, d1_data, ddims, *d1_mask_args(more=d1_mdata_args)) / d1_num_samples - np.square(d1_mean)
    d1_var *= domain1[axis].flat

    if d2_mdata is None:
        d2_num_samples = np.prod(np.array([domain2[i].sum() if domain2[i].size>1 else d2_data.shape[i] for i in sample_axes]))
        d2_mdata_args = []
    else:
        d2_num_samples = np.einsum(d2_mdata, ddims, *d2_mask_args(), dtype=np.int)
        d2_mdata_args = [d2_mdata, range(0, ndim)]
    
    d2_mean = np.einsum(d2_data, ddims, *d2_mask_args(more=d2_mdata_args)) / d2_num_samples
    d2_mean *= domain2[axis].flat
    d2_var = np.einsum(d2_data, ddims, d2_data, ddims, *d2_mask_args(more=d2_mdata_args)) / d2_num_samples - np.square(d2_mean)
    d2_var *= domain2[axis].flat

    t1 = np.square(d1_var / d1_num_samples) / (d1_num_samples - 1)
    t2 = np.square(d2_var / d2_num_samples) / (d2_num_samples - 1)
    degrees_freedom = np.square(d1_var / d1_num_samples + d2_var / d2_num_samples) / (t1 + t2)

    t_stat = np.abs(np.divide(d1_mean - d2_mean, np.sqrt(d1_var / d1_num_samples + d2_var / d2_num_samples)))

    p_val = t.sf(t_stat, degrees_freedom) / domain1[axis].flat # intentional divide by zero
    fold_change = np.divide(d1_mean, d2_mean) / domain1[axis].flat # intentional divide by zero
    return p_val, fold_change

def _local_search(search_function, darr, seed, axis, global_query, marr, mseed, **kwargs):
    import numpy as np
    query, data, mdata, empty = _get_local_query(darr, marr, global_query)
    # skip computation and return sensible outputs when some axes of the query are empty
    if len(empty)>0:
        if axis in empty:
            return np.array([], dtype=darr.ndarray.dtype)
        else:
            result = np.empty((darr.local_shape[axis]), dtype=darr.ndarray.dtype)
            result.fill(np.nan)
            return result

    return globals()[search_function](data, seed, axis, query, mdata, mseed, **kwargs)

# apply a query to local chunks of the datacube. boolean selectors just need to be chunked in the same
# way as the datacube, while integer indices need to be individually mapped.  for integer indices,
# the selected part of the datacube chunk is copied to a new array.  if there are no integer indices,
# a view on the local array is returned.
def _get_local_query(darr, marr, global_query):
    import numpy as np
    assert(len(global_query)==darr.ndim)
    
    local_query = [np.ones(1, dtype=darr.dtype)]*darr.ndim
    local_subscripts = [range(0,darr.local_shape[axis]) for axis in range(0,darr.ndim)]
    subscripted = False
    empty = []
    for axis, mask in enumerate(global_query):
        if isinstance(mask, np.ndarray):
            if mask.dtype == np.bool and mask.size>0:
                local_query[axis] = mask[darr.distribution._maps[axis].global_slice].astype(darr.dtype)
            elif np.issubdtype(mask.dtype, np.integer):
                subscripted = True
                local_indices = []
                for global_index in mask:
                    try:
                        local_index = darr.distribution._maps[axis].local_from_global_index(global_index)
                        local_indices.append(local_index)
                    except IndexError:
                        continue
                
                if len(local_indices)>0:
                    local_subscripts[axis] = local_indices
                else:
                    empty.append(axis)
            else:
                raise error.DatacubeSelectorError('Query selector must be either boolean or integer ndarray, not %s' % str(mask.dtype))

    mdata = None
    if len(empty)>0:
        data = None
    elif subscripted:
        data = darr.ndarray[np.ix_(*local_subscripts)]
        if marr is not None:
            mdata = marr.ndarray[np.ix_(*local_subscripts)]
    else:
        data = darr.ndarray
        if marr is not None:
            mdata = marr.ndarray

    return local_query, data, mdata, empty


class DatacubeCore:
    def __init__(self, data, observed=None, distributed=False, distribution=None):
        assert(isinstance(data, np.ndarray))
        self.data = data
        self.ndim = data.ndim
        self.shape = data.shape
        self.size = data.size
        self.dtype = data.dtype
        if observed is not None:
            assert(isinstance(observed, np.ndarray))
            assert(observed.shape == data.shape)
            self.data[np.isnan(self.data)] = 0.0 # make sure unobserved entries are not NaN
        self.observed = observed

        # precompute and store some statistics on the datacube
        self.mean = []
        self.std = []
        for axis in range(0, self.ndim):
            self.mean.append(np.nanmean(self.data, axis, keepdims=True))
            self.std.append(np.nanstd(self.data, axis, keepdims=True))

        # set up for distributed computations
        self.distributed = distributed
        if self.distributed:
            import distarray
            from distarray.globalapi import Context, Distribution
            from ipyparallel.error import CompositeError

            self.context = Context()
            if distribution is not None:
                assert(len(distribution) == data.ndim)
                self.distribution = Distribution(self.context, self.data.shape, dist=distribution)
            else:
                # default to block-distributed on first axis and not distributed otherwise
                # TODO: support other distributions
                self.distribution = Distribution(self.context, self.data.shape, tuple('b' if i == 0 else 'n' for i in range(0,data.ndim)))

            self.dist_data = self.context.fromarray(self.data, self.distribution)
            if self.observed is not None:
                self.dist_observed = self.context.fromarray(self.observed, self.distribution)
            else:
                self.dist_observed = None

            self.context.push_function(_get_local_query.__name__, _get_local_query)
            self.context.push_function(_mask_args.__name__, _mask_args)
            self.context.push_function(correlation.__name__, correlation)
            self.context.push_function(fold_change.__name__, fold_change)
            self.context.push_function(differential.__name__, differential)

    def get_data(self, subscripts):
        return self.data[subscripts]

    def get_log2_1p(self, subscripts):
        return np.log1p(self.data[subscripts])/np.log(2.0)

    def get_standard(self, subscripts, axis):
        sample_subs = np.ix_(*[[0] if i == axis else x.flat for i, x in enumerate(subscripts)])
        return (self.data[subscripts] - self.mean[axis][sample_subs]) / self.std[axis][sample_subs]

    """Compute correlation for each sample versus a particular seed index, along an axis of this datacube.

    Restricted indices along any axis (including the query axis) can be selected, by supplying either a
    1-dimensional array of indices or a boolean array, per axis (or numpy.ones(1) to select all for that
    axis).

    The choice between integer and boolean interfaces implies a time/memory trade-off. For all axes using
    integer indices, a smaller array will be copied out from the full data, costing additional memory but
    saving compute time on omitted entries. For any remaining axes using boolean masks, the computation is
    done in place, costing no memory but "wasting" time on omitted entries by carrying out a calculation
    that doesn't contribute to the final result.

    Args:
        seed_index (int): Index along `axis` containing seed sample against which to compare.
        axis (int): Axis to operate along, in range(0, self.ndim)
        query (Optional[list]): list of length data.ndim, where the ith element is either numpy.ones(1)
            ("all"), a 1-dimensional array of integral indices, or a boolean array of shape=(data.shape[i],),
            where True indicates that the values at that index should be included.

    Returns:
        numpy.ndarray: Array of shape (data.shape[axis],), or (query[axis].size,) if using integral indices
            along the query axis.
    """
    # TODO: automatically convert to boolean vs integer indexes to optimize memory/speed
    def get_correlated(self, seed_index, axis, query=None):
        if query is None:
            query = [None]*self.ndim

        seed_observed = None
        if self.observed is not None:
            seed_observed = self.observed.take([seed_index], axis=axis)

        seed = self.data.take([seed_index], axis=axis)
        for i, mask in enumerate(query):
            if i != axis and mask is not None and mask.dtype != np.bool and np.issubdtype(mask.dtype, np.integer):
                seed = seed.take(mask, axis=i)
                if seed_observed is not None:
                    seed_observed = seed_observed.take(mask, axis=i)
        
        if self.distributed:
            try:
                assert(all(d == 'n' for i, d in enumerate(self.distribution.dist) if i != axis)) # TODO: correlation_search currently needs all non-query axes to be non-distributed
                dist_observed_key = self.dist_observed.key if self.dist_observed is not None else None
                args = (correlation.__name__, self.dist_data.key, seed, axis, query, dist_observed_key, seed_observed)
                return np.concatenate(self.dist_data.context.apply(_local_search, args))
            except CompositeError as e:
                raise error.ParallelError(self._format_composite_error(e))
        else:
            selected_data, selected_observed, query = self._apply_query(query)
            return correlation(selected_data, seed, axis, query, selected_observed, seed_observed)

    def get_fold_change(self, axis, domain1=None, domain2=None):
        if domain1 is None:
            domain1 = [np.ones(1, dtype=self.dtype)]*self.ndim
        if domain2 is None:
            domain2 = [np.ones(1, dtype=self.dtype)]*self.ndim

        if self.distributed:
            try:
                assert(all(d == 'n' for i, d in enumerate(self.distribution.dist) if i != axis))
                dist_observed_key = self.dist_observed.key if self.dist_observed is not None else None
                args = (fold_change.__name__, self.dist_data.key, axis, domain1, domain2, dist_observed_key)
                return np.concatenate(self.dist_data.context.apply(_local_differential, args))
            except CompositeError as e:
                raise error.ParallelError(self._format_composite_error(e))
        else:
            domain2[axis] = domain1[axis]
            d1_data, d1_mdata, domain1 = self._apply_query(domain1)
            d2_data, d2_mdata, domain2 = self._apply_query(domain2)
            return fold_change(d1_data, d2_data, axis, domain1, domain2, d1_mdata, d2_mdata)

    def get_differential(self, axis, domain1=None, domain2=None):
        if domain1 is None:
            domain1 = [np.ones(1, dtype=self.dtype)]*self.ndim
        if domain2 is None:
            domain2 = [np.ones(1, dtype=self.dtype)]*self.ndim

        if self.distributed:
            try:
                assert(all(d == 'n' for i, d in enumerate(self.distribution.dist) if i != axis))
                dist_observed_key = self.dist_observed.key if self.dist_observed is not None else None
                args = (differential.__name__, self.dist_data.key, axis, domain1, domain2, dist_observed_key)
                result = self.dist_data.context.apply(_local_differential, args)
                return tuple(np.concatenate([r[i] for r in result]) for i in range(0, len(result[0])))
            except CompositeError as e:
                raise error.ParallelError(self._format_composite_error(e))
        else:
            domain2[axis] = domain1[axis]
            d1_data, d1_mdata, domain1 = self._apply_query(domain1)
            d2_data, d2_mdata, domain2 = self._apply_query(domain2)
            return differential(d1_data, d2_data, axis, domain1, domain2, d1_mdata, d2_mdata)

    # error from parallel nodes needs special call to render_traceback()
    # and comes back with ANSI control codes which need to be stripped
    def _format_composite_error(self, e):
        import re
        ansi_escape = re.compile(r'(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]')
        tb = '\n'.join([ansi_escape.sub('', x) for x in e.render_traceback()])
        print(tb)
        # TODO: provide information about the compute nodes where the error(s) happened
        return tb

    # apply integer indexes in query by copying the subarray to a new array,
    # and updating the query accordingly. also convert None selectors to
    # the trivial mask array [1.0]
    def _apply_query(self, query):
        data = self.data
        mdata = self.observed
        for i, mask in enumerate(query):
            if mask is not None and mask.dtype != np.bool and np.issubdtype(mask.dtype, np.integer):
                data = data.take(mask, axis=i)
                query[i] = np.ones(1, dtype=self.dtype)
                if self.observed is not None:
                    mdata = mdata.take(mask, axis=i)
            elif mask is None:
                query[i] = np.ones(1, dtype=self.dtype)
        return data, mdata, query
