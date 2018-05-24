import json
import collections

from twisted.python import log
import logging


SUMMARY_STATS_CALCULATORS = {
    'min': lambda dataset, field: dataset[field].min().values.tolist(),
    'max': lambda dataset, field: dataset[field].max().values.tolist(),
    'mean': lambda dataset, field: dataset[field].mean().values.tolist(),
    'std': lambda dataset, field: dataset[field].std().values.tolist(),
}

SUMMARY_STATS_CONDITIONS = {
    'min': lambda dataset, field: True,
    'max': lambda dataset, field: True,
    'mean': lambda dataset, field: not dataset[field].dtype.kind in 'OSU',
    'std': lambda dataset, field: not dataset[field].dtype.kind in 'OSU',
}


def calculate_summary_stats(dataset, calculators=None, conditions=None):
    ''' Calculate summary_statistics for each field in a Dataset.

    Parameters
    ----------
    dataset : xarray.Dataset
        Compute summary statistics for fields in this dataset.
    calculators : dict, optional
        Keys are string identifiers for a summary statistic type, values are functions taking a dataset and a field 
        which compute that statistic over the data in that field. Defaults to SUMMARY_STATS_CALCULATORS.
    conditions : dict, optional
        Keys are string identifiers for a summary statistic type, values are functions taking a dataset and a field 
        which deetermine whether that statistic should be computed for that field. Defaults to SUMMARY_STATS_CALCULATORS.

    Returns
    -------
    stats : dict
        Keys are string identifiers for summary statistic types. Values are dictionaries mapping fields to computed  
        values.

    '''

    if calculators is None:
        calculators = SUMMARY_STATS_CALCULATORS

    if conditions is None:
        conditions = SUMMARY_STATS_CONDITIONS

    stats = collections.defaultdict(dict, {})
    for field in dataset.variables:
        log.msg('calculating stats for field \'{}\'...'.format(field), logLevel=logging.INFO)

        for stat_type, stat_calculator in calculators.items():
            if not conditions[stat_type](dataset, field):
                continue

            stats[stat_type][field] = stat_calculator(dataset, field)
            log.msg('{} of field {} is: {}'.format(stat_type, field, stats[stat_type][field]), logLevel=logging.INFO)

    return stats


def write_stats_to_json(path, stats):
    with open(path, 'w') as json_file:
        json.dump(stats, json_file, indent=2)
    

def read_stats_from_json(path):
    with open(path, 'r') as json_file:
        return json.load(json_file)


def cache_summary_statistics(dataset, reader, writer, force=False, calculators=None, conditions=None):
    ''' Lazily compute summary statistics, reading from a cache if possible.

    Parameters
    ----------
    dataset : xarray.Dataset
        Compute summary statistics from this dataset's variables
    reader : function
        Reads a stats dictionary, as would be returned by calculate_summmary_statistics, from an external resource.
    writer : function
        Writes a stats dictionary to an external resource.
    force : bool, optional
        If True (default is False), stats will be calculated from dataset regardless of previous caching.
    calculators : dict, optional
        Keys are string identifiers for a summary statistic type, values are functions taking a dataset and a field 
        which compute that statistic over the data in that field. Defaults to SUMMARY_STATS_CALCULATORS.
    conditions : dict, optional
        Keys are string identifiers for a summary statistic type, values are functions taking a dataset and a field 
        which deetermine whether that statistic should be computed for that field. Defaults to SUMMARY_STATS_CALCULATORS.

    Returns
    -------
    stats : dict
        Keys are string identifiers for summary statistic types. Values are dictionaries mapping fields to computed  
        values.
        
    '''

    if calculators is None:
        calculators = SUMMARY_STATS_CALCULATORS

    if not force:
        try:
            stats = reader()

            for key in calculators.keys():
                assert key in stats

                for field in dataset.variables:
                    assert field in stats[key]
            
            return stats

        except (OSError, IOError, json.JSONDecodeError, AssertionError) as err:
            log.msg('Failed to extract summary statistics from file.\n {}'.format(repr(err)), logLevel=logging.WARNING)

    stats = calculate_summary_stats(dataset, calculators=calculators, conditions=conditions)
    writer(stats)
    
    return stats
    