import os
import json
import functools

import pytest
import pandas as pd
import requests


datacube_wamp_transport = os.environ.get('DATACUBE_WAMP_TRANSPORT')
datacube_wamp_realm = os.environ.get('DATACUBE_WAMP_REALM')
conn_bridge_port = os.environ.get('CONN_BRIDGE_PORT')
manifest_path = os.environ.get('MANIFEST_PATH')


@pytest.fixture(autouse=True)
def network_config_specified():

    if datacube_wamp_realm is None:
        pytest.skip(msg='no DATACUBE_WAMP_REALM specified')
    if datacube_wamp_transport is None:
        pytest.skip(msg='no DATACUBE_WAMP_TRANSPORT specified')
    if conn_bridge_port is None:
        pytest.skip(msg='no CONN_BRIDGE_PORT specified')
    global manifest_path
    if manifest_path is None:
        parent_dir = os.path.dirname(__file__)
        manifest_path = os.path.join(parent_dir, 'manifest.json')


def get_manifest_suite(suite):
    with open(manifest_path, 'r') as manifest_file:
        manifest = json.load(manifest_file)

    return [ item for item in manifest if item['suite'] == suite ]


@pytest.fixture(
    scope='module'
)
def load_query():
    def _load_query(entry):

        parent_dir = os.path.dirname(__file__)
        exp_path = os.path.join(parent_dir, 'data', '{}.{}'.format(
            entry['name'],
            entry['extension']
        ))

        with open(exp_path, 'r') as exp_file:
            expected = json.load(exp_file)

        query_string = 'http://localhost:{}/{}'.format(conn_bridge_port, entry['query'])
        obtained = requests.get(query_string).json()

        return expected, obtained

    return _load_query


def compare_response_keys(expected, obtained):
    assert( set(expected.keys()) == set(obtained.keys()) )
    for key, value in expected.items():
        if key != 'msg':
            assert( obtained[key] == value )


def compare_response_columns(expected, obtained):

    expected = pd.DataFrame(expected['msg'])
    obtained = pd.DataFrame(obtained['msg'])
    
    assert( set(obtained.columns.values) == set(expected.columns.values) )


def compare_response_values(expected, obtained):
    expected = pd.DataFrame(expected['msg'])
    obtained = pd.DataFrame(obtained['msg'])

    print('==================== expected ====================')
    print(expected.head())
    print('==================== obtained ====================')
    print(obtained.head())
    
    obtained = obtained[expected.columns.values]
    pd.testing.assert_frame_equal( obtained, expected )


@pytest.fixture(
    params=[
        compare_response_keys,
        compare_response_columns, 
        compare_response_values
    ]
)
def compare_responses(request):
    return request.param


@pytest.fixture(
    params=get_manifest_suite('correlation'),
    scope='module'
)
def correlation_query(load_query, request):
    return load_query(request.param), request.param


@pytest.fixture(
    params=get_manifest_suite('structure'),
    scope='module'
)
def structure_query(load_query, request):
    return load_query(request.param), request.param


@pytest.fixture(
    params=get_manifest_suite('spatial'),
    scope='module'
)
def spatial_query(load_query, request):
    return load_query(request.param), request.param


@pytest.fixture(
    params=get_manifest_suite('injection_coordinate'),
    scope='module'
)
def injection_coordinate_query(load_query, request):
    return load_query(request.param), request.param
