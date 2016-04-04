import numpy as np
import scipy as sp
import distarray
from distarray.globalapi import Context, Distribution

def correlation_search(data, seed, axis, query=None, mdata=None, mseed=None):
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
        query = [np.ones(1)]*data.ndim

    mask_args = \
        lambda axes=range(0,data.ndim), out_axes=[axis], more=[]: \
            list(sum([(mask.squeeze(),[i]) for i, mask in enumerate(query) if i in axes and mask.size>1], ()))+more+[out_axes]
    
    sdims = range(0,seed.ndim)
    ddims = range(0,data.ndim)
    sample_axes = np.array([i for i in ddims if i != axis])

    if mdata is None:
        seed_num_samples = np.prod(np.array([query[i].sum() if query[i].size>1 else data.shape[i] for i in sample_axes]))
        num_samples = seed_num_samples
        mseed_args = []
        mdata_args = []
    else:
        seed_num_samples = np.einsum(mseed, range(0,mseed.ndim), *mask_args(sample_axes, []))
        num_samples = np.einsum(mdata, range(0,mdata.ndim), *mask_args())
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
    denominator += np.sum(data_mean**2, axis=sample_axes) * num_samples
    denominator *= np.einsum(seed_dev**2, sdims, *mask_args(sample_axes))
    denominator = np.sqrt(denominator)

    return np.clip(numerator / denominator, -1.0, 1.0) / query[axis].flat # intentional divide by zero


def _local_search(search_function, darr, seed, axis, global_query, marr, mseed):
    import numpy as np
    query, data, mdata, empty = _get_local_query(darr, marr, global_query)
    if len(empty)>0:
        if axis in empty:
            return np.array([], dtype=darr.ndarray.dtype)
        else:
            result = np.empty((darr.local_shape[axis]), dtype=darr.ndarray.dtype)
            result.fill(np.nan)
            return result

    return globals()[search_function](data, seed, axis, query, mdata, mseed)


def _get_local_query(darr, marr, global_query):
    import numpy as np
    assert(len(global_query)==darr.ndim)
    
    local_query = [np.ones(1)]*darr.ndim
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
                raise IndexError

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


class Datacube:
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
            assert(observed.dtype == data.dtype)
            self.data = self.data / observed # make sure unobserved entries are NaN
        self.observed = observed

        self.mean = []
        self.std = []
        for axis in range(0, self.ndim):
            self.mean.append(np.nanmean(self.data, axis, keepdims=True))
            self.std.append(np.nanstd(self.data, axis, keepdims=True))

        self.distributed = distributed
        if self.distributed:
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
            self.context.push_function(correlation_search.__name__, correlation_search)

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
    def get_correlated(self, seed_index, axis, query=None):
        if query is None:
            query = [np.ones(1, dtype=self.dtype)]*self.ndim

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
            assert(all(d == 'n' for i, d in enumerate(dist) if i != axis)) # TODO: correlation_search currently needs all non-query axes to be non-distributed
            dist_observed_key = self.dist_observed.key if self.dist_observed is not None else None
            args = (correlation_search.__name__, self.dist_data.key, seed, 0, query, dist_observed_key, seed_observed)
            return np.concatenate(self.dist_data.context.apply(_local_search, args))
        else:
            selected_data = self.data
            selected_observed = self.observed
            for i, mask in enumerate(query):
                if mask is not None and mask.dtype != np.bool and np.issubdtype(mask.dtype, np.integer):
                    selected_data = selected_data.take(mask, axis=i)
                    query[i] = np.ones(1, dtype=self.dtype)
                    if self.observed is not None:
                        selected_observed = selected_observed.take(mask, axis=i)
            
            return correlation_search(selected_data, seed, axis, query, selected_observed, seed_observed)
