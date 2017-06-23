import numpy as np
from scipy.stats import rankdata
import json
import pickle
import redis
#import txredisapi
#from twisted.internet import reactor

#todo: use xr.Dataset instead of np structured array?
#import xarray as xr
#
#def np_structured_array_to_xr_dataset(sa):
#    return xr.Dataset({field: ('dim_0', sa[field]) for field in sa.dtype.names})

class Datacube:


    def __init__(self, npy_file=None):
        if npy_file: self.load(npy_file)
        #todo: would be nice to find a way to swap these out
        #if reactor.running:
        #    self.redis_client = txredisapi.Connection('localhost', 6379)
        #else:
        self.redis_client = redis.StrictRedis('localhost', 6379)


    def load(self, npy_file):
        #todo: rename df
        #todo: argsorts need to be cached to a file
        self.df = np.load(npy_file, mmap_mode='r')
        self.argsorts = {}
        for field in self.df.dtype.names:
            self.argsorts[field] = np.argsort(self.df[field])


    def select(self,
               filters=None,
               sort=None,
               ascending=None,
               start=0,
               stop=None,
               indexes=None,
               fields=None,
               options={}):
        r = self.df
        redis_client = self.redis_client
        if not filters and not fields == "indexes_only":
            if indexes:
                num_results = np.array(indexes)[start:stop].size
            else:
                num_results = r['index'][start:stop].size
            if options.get('max_records') and num_results > options.get('max_records'):
                raise ValueError('Requested would return ' + str(num_results) + ' records; please limit request to ' + str(options['max_records']) + ' records.')
        inds = np.array(range(r.size), dtype=np.int)
        if filters is not None:
            inds = self._query(filters)
        if indexes is not None:
            indexes = [x for x in indexes if x != None]
            if inds is not None:
                inds = inds[np.in1d(inds, indexes)]
            else:
                inds = np.array(indexes, dtype=np.int)
        if sort is not None and r.size > 0:
            sorted_inds = self._sort(sort, ascending)
            if inds is not None:
                inds = sorted_inds[np.in1d(sorted_inds, inds)]
            else:
                inds = sorted_inds
        inds = inds[start:stop]
        if fields == "indexes_only":
            return inds
        else:
            if options.get('max_records') and inds.size > options.get('max_records'):
                raise ValueError('Requested would return ' + str(inds.size) + ' records; please limit request to ' + str(options['max_records']) + ' records.')
            if fields and type(fields) is list:
                for field in fields:
                    if field not in r.dtype.names:
                        raise ValueError('Requested field \'' + str(field) + '\' does not exist. Allowable fields are: ' + str(r.dtype.names))
                res = r[inds][fields]
            else:
                res = r[inds]
            return res


    def _sort(self, sort, ascending):
        df = self.df
        redis_client = self.redis_client
        #todo: possibly should reincorporate caching
        if not ascending:
            ascending = [True] * len(sort)
        for field in sort:
            if field not in df.dtype.names:
                raise ValueError('Requested sort field \'' + str(field) + '\' does not exist. Allowable fields are: ' + str(df.dtype.names))
        s = df[sort]
        ranks = np.zeros((s.size, len(s.dtype)), dtype=np.int)
        for idx, (field, asc) in enumerate(zip(sort[::-1], ascending[::-1])):
            if s[field].dtype.name.startswith('string') or s[field].dtype.name.startswith('unicode'):
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
        return res


    def _query(self, filters):
        df = self.df
        redis_client = self.redis_client
        column_argsort = self.argsorts
        if not filters:
            return np.array(range(df.size), dtype=np.int)
        else:
            # todo: add name to datacube object and include in cache keys

            def _do_search(f):
                op = f['op']
                field = f['field']
                value = f['value']

                res = np.zeros(df[field].shape, dtype=np.bool)
                start = 0
                stop = df[field].size
                if op == '=' or op == 'is':
                    start = np.searchsorted(df[field], value, side='left', sorter=column_argsort[field])
                    stop = np.searchsorted(df[field], value, side='right', sorter=column_argsort[field])
                elif op == '<':
                    stop = np.searchsorted(df[field], value, side='left', sorter=column_argsort[field])
                elif op == '>':
                    start = np.searchsorted(df[field], value, side='right', sorter=column_argsort[field])
                elif op == '<=':
                    stop = np.searchsorted(df[field], value, side='right', sorter=column_argsort[field])
                elif op == '>=':
                    start = np.searchsorted(df[field], value, side='left', sorter=column_argsort[field])
                res[column_argsort[field][start:stop]] = True
                return res

            expanded_filters = []
            for f in filters:
                op, field, value = (f['op'], f['field'], f['value'])
                if op == 'between':
                    expanded_filters.append({'op': '>=', 'field': field, 'value': value[0]})
                    expanded_filters.append({'op': '<=', 'field': field, 'value': value[1]})
                elif op == 'in':
                    or_clause = []
                    for v in value:
                        or_clause.append({'op': '=', 'field': field, 'value': v})
                    expanded_filters.append(or_clause)
                else:
                    expanded_filters.append(f)

            def _filter_cache_key(f):
                return json.dumps(['filter', f['op'], f['field'], f['value']])

            def _pack(bool_array):
                return str(np.packbits(bool_array).tobytes())

            def _unpack(bitstring, sz):
                res = np.fromstring(bitstring, dtype=np.uint8)
                res = res[:sz]
                return res

            def _cache_filter(f):
                filter_cache_key = _filter_cache_key(f)
                cached = redis_client.exists(filter_cache_key)
                if not cached:
                    filtered = _do_search(f)
                    redis_client.setnx(filter_cache_key, _pack(filtered))
                return filter_cache_key

            and_keys = []
            for f in expanded_filters:
                if isinstance(f, list):
                    or_keys = []
                    for o in f:
                        key = _cache_filter(o)
                        or_keys.append(key)
                    or_cache_key = json.dumps(['filter_or', or_keys])
                    lua = '''
                        local exists=redis.call('EXISTS', KEYS[1])
                        if 0 == exists then
                            redis.call('BITOP', 'OR', unpack(KEYS))
                        end
                        return redis.call('GET', KEYS[1])
                    '''
                    eval_keys = (or_cache_key,)+tuple(or_keys)
                    redis_client.eval(lua, len(eval_keys), *eval_keys)
                    and_keys.append(or_cache_key)
                else:
                    key = _cache_filter(f)
                    and_keys.append(key)

            and_cache_key = json.dumps(['filter_and', and_keys])
            lua = '''
                local exists=redis.call('EXISTS', KEYS[1])
                if 0 == exists then
                    redis.call('BITOP', 'AND', unpack(KEYS))
                end
                return redis.call('GET', KEYS[1])
            '''
            eval_keys = (and_cache_key,)+tuple(and_keys)
            res = redis_client.eval(lua, len(eval_keys), *eval_keys)
            res = _unpack(res, df.size)
            res = np.flatnonzero(res)
            return res
