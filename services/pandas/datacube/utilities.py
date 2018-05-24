
import collections

from twisted.python import log
import logging


SUMMARY_STATS_CALCULATORS = {
    'min': lambda dataset, field: dataset[field].min().values,
    'max': lambda dataset, field: dataset[field].max().values,
    'mean': lambda dataset, field: dataset[field].mean().values if not dataset[field].dtype.kind in 'OSU' else None,
    'std': lambda dataset, field: dataset[field].std().values if not dataset[field].dtype.kind in 'OSU' else None,
}


def calculate_summary_stats(dataset, calculators=None):
    ''' Calculate summary_statistics for each field in a Dataset.
    '''

    if calculators is None:
        calculators = SUMMARY_STATS_CALCULATORS

    stats = collections.defaultdict(dict, {})
    for field in dataset.variables:
        log.msg('calculating stats for field \'{}\'...'.format(field), logLevel=logging.INFO)

        for stat_type, stat_calculator in calculators.items():
            stats[stat_type][field] = stat_calculator(dataset, field)
            log.msg('{} of field {} is: {}'.format(stat_type, field, stats[stat_type][field]), logLevel=logging.INFO)

    return stats


def write_stats_to_json(stats, path):
    with open(path, 'w') as json_file:
        json.dump(stats, json_file, indent=2)
    

def read_stats_from_json(path):
    with open(path, 'r') as json_file:
        return json.load(json_file)


def cache_summary_statistics(dataset, reader=None, writer=None, calculators=None):

    if calculators is None:
        calculators = SUMMARY_STATS_CALCULATORS

    try:
        stats = reader()

        for key in calculators.keys():
            assert key in stats

            for field in dataset.variables:
                assert field in stats[key]

    except (OSError, IOError, AssertionError):
        stats = calculate_summary_stats(dataset, calculators=calculators)
        writer(stats)
    
    return stats
    
