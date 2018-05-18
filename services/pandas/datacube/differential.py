import numpy as np
import scipy.stats


def differential(d1_data, d2_data, axis, domain1=None, domain2=None, d1_mdata=None, d2_mdata=None):
    '''

    Notes
    -----
    not currently in use

    '''

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

    p_val = scipy.stats.t.sf(t_stat, degrees_freedom) / domain1[axis].flat # intentional divide by zero
    fold_change = np.divide(d1_mean, d2_mean) / domain1[axis].flat # intentional divide by zero
return p_val, fold_change