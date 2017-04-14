import numpy as np
from datacube_core import DatacubeCore
import error

class Datacube(DatacubeCore):
    def info(self):
        return {
            'ndim': self.ndim,
            'shape': list(self.shape),
            'dtype': self.dtype.name
        }

    def raw(self, select):
        self._validate_select(select)
        subscripts = self._get_subscripts_from_select(select)
        return self.get_data(subscripts)

    def log2_1p(self, select):
        self._validate_select(select)
        subscripts = self._get_subscripts_from_select(select)
        return self.get_log2_1p(subscripts)

    def standard(self, select, axis):
        self._validate_select(select)
        subscripts = self._get_subscripts_from_select(select)
        return self.get_standard(subscripts, axis)

    def corr(self, select, axis, seed):
        self._validate_select(select)
        query = self._parse_select(select)
        query = self._convert_slices_to_indices(query)

        r=self.get_correlated(seed, axis, query)
        sort_idxs = np.argsort(-r)
        return {'indexes': sort_idxs.tolist(), 'correlations': [None if np.isnan(x) else float(x) for x in r]}

    def fold_change(self, numerator, denominator, axis):
        self._validate_select(numerator)
        self._validate_select(denominator)
        domain1 = self._parse_select(numerator)
        domain2 = self._parse_select(denominator)
        domain1 = self._convert_slices_to_indices(domain1)
        domain2 = self._convert_slices_to_indices(domain2)

        r=self.get_fold_change(axis, domain1, domain2)
        r[np.logical_not(np.isfinite(r))] = np.nan
        sort_idxs = np.argsort(-r)
        return {'indexes': sort_idxs.tolist(), 'fold_changes': [None if np.isnan(x) else float(x) for x in r]}

    def diff(self, numerator, denominator, axis):
        self._validate_select(numerator)
        self._validate_select(denominator)
        domain1 = self._parse_select(numerator)
        domain2 = self._parse_select(denominator)
        domain1 = self._convert_slices_to_indices(domain1)
        domain2 = self._convert_slices_to_indices(domain2)

        p_vals, fold_changes = self.get_differential(axis, domain1, domain2)
        p_vals[np.logical_not(np.isfinite(p_vals))] = np.nan
        sort_idxs = np.argsort(p_vals)
        return {'indexes': sort_idxs.tolist(),
                'p_values': [None if np.isnan(x) else float(x) for x in p_vals],
                'fold_changes': [None if np.isnan(x) else float(x) for x in fold_changes]}


    # interpret strings in selector as restful service uri's
    # and insert the json response in their place.
    def _selector_service_calls(self, select):
        for axis, selector in enumerate(select):
            if isinstance(selector, basestring):
                try:
                    select[axis] = json.load(urllib2.urlopen(selector))
                except URLError as e:
                    raise error.ServiceError(str(e.reason))
        return select

    def _validate_select(self, select):
        if len(select) != self.ndim:
            raise error.SelectorError('Number of select elements must match number of dimensions of the datacube ({0}).'.format(self.ndim))

        for axis, selector in enumerate(select):
            if isinstance(selector, list):
                if any(type(x) != type(selector[0]) for x in selector):
                    raise error.SelectorError('All elements of selector for axis {0} do not have the same type.'.format(axis))
                if not isinstance(selector[0], (int, long)) and not isinstance(selector[0], bool):
                    raise error.SelectorError('Elements of list selector for axis {0} must be of type int or bool.'.format(axis))
                if isinstance(selector[0], bool) and len(selector) != self.shape[axis]:
                    raise error.SelectorError('Boolean list selector for axis {0} must have length {1} to match the size of the datacube.'.format(axis, self.shape[axis]))
            elif isinstance(selector, dict):
                keys = ['start', 'stop', 'step']
                for key in keys:
                    if key in selector and selector[key] is not None and not isinstance(selector[key], (int, long)):
                        raise error.SelectorError('Slice selector for axis {0} must have ''{1}'' of type int.'.format(axis, key))

    # convert dict selectors into slice objects,
    # index and bool ndarrays
    def _parse_select(self, select_element):
        select = [slice(None,None,None)]*self.ndim
        assert(isinstance(select_element, list))
        assert(len(select_element) == self.ndim)
        for axis, selector in enumerate(select_element):
            if isinstance(selector, list):
                if len(selector) == 0:
                    select[axis] = np.array([], dtype=np.int)
                elif all(type(x) == bool for x in selector):
                    select[axis] = np.array(selector, dtype=np.bool)
                elif all(isinstance(x, (int, long)) for x in selector):
                    select[axis] = np.array(selector, dtype=np.int)
                else:
                    raise error.SelectorError('All elements of selector for axis {0} do not have the same type.'.format(axis))
            elif isinstance(selector, dict):
                select[axis] = slice(selector.get('start'), selector.get('stop'), selector.get('step'))
        return select

    # parse and convert all request selectors into index arrays
    def _get_subscripts_from_select(self, select_element):
        subscripts = self._parse_select(select_element)
        for axis, subs in enumerate(subscripts):
            if isinstance(subs, np.ndarray) and subs.dtype == np.bool:
                subscripts[axis] = subs.nonzero()[0]
            elif isinstance(subs, slice):
                subscripts[axis] = np.array(range(*subs.indices(self.shape[axis])), dtype=np.int)
        subscripts = np.ix_(*subscripts)
        return subscripts

    # slices are materialized as 1-d arrays of integer indices, except
    # slice(None,None,None) which is encoded as None. boolean and integer
    # selectors are left as-is.
    def _convert_slices_to_indices(self, query):
        for axis, subs in enumerate(query):
            if isinstance(subs, slice):
                if subs == slice(None,None,None):
                    query[axis] = None
                else:
                    query[axis] = np.array(range(*subs.indices(self.shape[axis])), dtype=np.int)
        return query
