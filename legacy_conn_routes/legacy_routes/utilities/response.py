import functools

import pandas as pd
import numpy as np


def add_json_headers(request):
    request.responseHeaders.addRawHeader(b'content-type', b'application/json')
    request.responseHeaders.addRawHeader(b'Access-Control-Allow-Origin', b'*')


def add_csv_headers(request, name):
    request.responseHeaders.addRawHeader(b'content-type', b'test/csv')
    request.responseHeaders.addRawHeader(b'content-disposition', 'attachment;filename={}.csv'.format(name).encode())
    request.responseHeaders.addRawHeader(b'Access-Control-Allow-Origin', b'*')


def dc_to_df(dc_response, postprocess_df=None):

    if postprocess_df is None:
        postprocess_df = lambda arg: arg 

    df = reorient_datacube_response(dc_response)
    df = postprocess_df(df)

    return df


def reorient_datacube_response(response, process_df=None):
    
    if process_df is None:
        process_df = lambda a: a
    
    df = {}

    for data_key, data_val in response['data_vars'].items():
        df[data_key] = data_val['data']

    df = pd.DataFrame.from_dict(df, orient='columns')

    if df.empty:
        return {}

    df = process_df(df)
    return df.to_dict('record')


def package_json_response(response):

    return {
        'msg': response,
        'numRows': len(response),
        'startRow': 0, # TODO: do this right, if possible
        'totalRows': len(response), # TODO: do this right, if possible,
        'success': 'true' # TODO: ok....
    }


def postprocess_injection_coordinates(df):

    make_injection_coordinates = lambda row: [
        int(np.around(row['injection_x'])), 
        int(np.around(row['injection_y'])), 
        int(np.around(row['injection_z']))
    ]

    df['injection-coordinates'] = df.apply( make_injection_coordinates, axis=1)

    df = df.drop(columns=['injection_x', 'injection_y', 'injection_z'])

    return df


def postprocess_injection_structure(row, ccf_store=None):

    sids = []

    for sid in row['injection_structures'].split('/'):
        if sid == '':
            continue

        sid = int(sid)
        if ccf_store is not None:
            sid = ccf_store.id_summary_map[sid]

        sids.append(sid)

    return sids



def postprocess_injection_structures(df, ccf_store=None):
    df['injection_structures'] = df.apply(
        functools.partial(postprocess_injection_structure, ccf_store=ccf_store),
        axis=1
    )

    return df