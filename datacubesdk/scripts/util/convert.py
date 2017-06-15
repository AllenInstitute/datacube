import pandas as pd
#import xarray as xr
import numpy as np


def pd_dataframe_to_np_structured_array(df):
    if 'index' in list(df):
        col_data = []
        col_names = []
        col_types = []
    else:
        col_data = [df.index]
        col_names = ['index']
        col_types = ['i4']
    for name in df.columns:
        column = df[name]
        data = np.array(column)

        if data.dtype.kind == 'O':
            if all(isinstance(x, basestring) or x is np.nan or x is None for x in data):
                data[data == np.array([None])] = b''
                data[np.array([True if str(x) == 'nan' else False for x in data], dtype=np.bool)] = b''
                data = np.array([x + '\0' for x in data], dtype=np.str)
        col_data.append(data)
        col_names.append(name)
        # javascript cannot natively handle longs
        if str(data.dtype) == 'int64':
            col_types.append('i4')
        elif str(data.dtype) == 'uint64':
            col_types.append('u4')
        else:
            col_types.append(data.dtype.str)
    out = np.array([tuple(data[j] for data in col_data) for j in range(len(df.index))],
                  dtype=[(str(col_names[i]), col_types[i]) for i in range(len(col_names))])
    return out


#def np_structured_array_to_xr_dataset(sa):
#    return xr.Dataset({field: ('dim_0', sa[field]) for field in sa.dtype.names})
