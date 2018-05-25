import functools
import unittest.mock as mock

import pytest
import xarray as xr
import numpy as np

from .. import summary_statistics


@pytest.fixture
def a_dataset():

    some_data = np.arange(30).reshape((5, 6))

    return xr.Dataset(
        {
            'some_data': (['an_axis', 'another_axis'], some_data)
        }
    )


def test_summary_statistics(a_dataset):

    obt = summary_statistics.calculate_summary_stats(a_dataset)

    assert( obt['min']['some_data'] == 0 )
    assert( obt['max']['some_data'] == 29 )
    assert( obt['mean']['some_data'] == np.mean(np.arange(30)) )
    assert( obt['std']['some_data'] == np.std(np.arange(30)) )


def test_summary_statistics_exclude(a_dataset):

    obt = summary_statistics.calculate_summary_stats(a_dataset, conditions={'mean': lambda *a, **k: False})

    assert( not 'mean' in obt )
    assert( obt['max']['some_data'] == 29 )


@pytest.mark.xfail(reason='mock patch fails on this relative import') #TODO
def test_cache_summary_stats(tmpdir_factory, a_dataset):

    file_name = 'basic_cache_test.json'
    file_path = tmpdir_factory.mktemp('data').join(file_name)

    write = functools.partial(summary_statistics.write_stats_to_json, str(file_path))
    read = functools.partial(summary_statistics.read_stats_from_json, str(file_path))

    first = summary_statistics.cache_summary_statistics(
        a_dataset,
        reader=read,
        writer=write,
        force=True
    )

    with mock.patch('summary_statistics.calculate_summary_stats') as p:

        second = summary_statistics.cache_summary_statistics(
            a_dataset,
            reader=read,
            writer=write,
            force=False
        )

        p.assert_not_called()

    assert( first['max']['some_data'] == 29 )
    assert( first['max']['some_data'] == second['max']['some_data'] )

