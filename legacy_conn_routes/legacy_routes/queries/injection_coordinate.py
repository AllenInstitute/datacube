import numpy as np

from legacy_routes.clauses.injection_structures import build_injection_structures_clause
from legacy_routes.clauses.transgenic_lines import build_transgenic_lines_clause
from legacy_routes.clauses.products import build_product_clause
from legacy_routes.utilities.response import dc_to_df, postprocess_injection_coordinates


INJECTION_COORDINATE_DETAILED_FIELDS = [
    'data_set_id',
    'specimen_name',
    'structure_id',
    'structure_abbrev',
    'structure_color',
    'strain',
    'transgenic_line',
    'gender',
    'injection_x',
    'injection_y',
    'injection_z',
    'injection_volume',
    'injection_structures',
    'product_id',
]

INJECTION_COORDINATE_DEFAULT_FIELDS = [
    'data_set_id',
    'injection_x',
    'injection_y',
    'injection_z',
]


def get_injection_coordinate_kwargs(
    seedPoint=None,
    injectionDistanceThreshold=None,
    transgenic_lines=None,
    injection_structures=None,
    primary_structure_only=True,
    product_ids=None,
    showDetail=False,
    startRow=0,
    numRows='all'
    ):

    filters = []

    if seedPoint is not None and injectionDistanceThreshold is not None:
        filters.append({
            op: 'distance',
            fields: ['injection_x', 'injection_y', 'injection_z'],
            point: seedPoint,
            value: injectionDistanceThreshold
        })

    if injection_structures is not None:
        filters.extend(build_injection_structures_clause(injection_structures, primary_structure_only))

    if transgenic_lines is not None:
        filters.extend(build_transgenic_lines_clause(transgenic_lines))

    if product_ids is not None:
        filters.extend(build_product_clause(product_ids))

    return {
        'fields': INJECTION_COORDINATE_DETAILED_FIELDS if showDetail else INJECTION_COORDINATE_DEFAULT_FIELDS,
        'coords': {},
        'select': {},
        'filters': filters
    }


def postprocess_injection_coordinates_search(df, seed, showDetail):

    df = postprocess_injection_coordinates(df)
    seed = np.array(seed)
    df['distance'] = df.apply(lambda row: np.linalg.norm(np.array(row['injection-coordinates']) - seed), axis=1)

    df = df.sort_values('distance', ascending=True)

    if not showDetail:
        df = df.drop(columns=['injection-coordinates'])
        return df

    df['num-voxels'] = None

    df = df.rename(columns={
        'injection_structures': 'injection-structures',
        'structure_name': 'structure-name',
        'injection_volume': 'injection-volume',
        'product_id': 'product-id',
        'structure_id': 'structure-id',
        'specimen_name': 'name',
        'structure_abbrev': 'structure-abbrev',
        'structure_color': 'structure-color',
        'transgenic_line': 'transgenic-line',
    })
    
    df['id'] = df['id'].astype(int)
    df['structure-id'] = df['structure-id'].astype(int)
    df['product-id'] = df['product-id'].astype(int)

    return df