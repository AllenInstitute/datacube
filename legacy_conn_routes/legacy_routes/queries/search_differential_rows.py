import numpy as np

from legacy_routes.clauses.domain import build_target_domain_clause, decode_domain_str, DEFAULT_TARGET_THRESHOLD
from legacy_routes.clauses.injection_structures import build_injection_structures_clause
from legacy_routes.clauses.transgenic_lines import build_transgenic_lines_clause
from legacy_routes.clauses.products import build_product_clause
from legacy_routes.utilities.response import dc_to_df, postprocess_injection_coordinates, postprocess_injection_structures


STRUCTURE_SEARCH_FIELDS_FULL = [
    "data_set_id", "experiment", "transgenic_line", "transgenic_line_id", "product_id", 
    "structure_id", "structure_abbrev", "structure_name", "specimen_name", "injection_volume", 
    "injection_structures", "injection_x", "injection_y", "injection_z", "gender", "strain", "volume",
    "structure_color"
] 


STRUCTURE_SEARCH_FIELDS_DEFAULT = [
    'data_set_id',
    'volume'
]


def nonoverlapping_structures_clause(structure_ids):
    return {
        'and': [
            {"field": "structure", "op": "in", "value": structure_ids},
            {
                "count": {
                    "field": "structures", 
                    "op": "in", 
                    "value": structure_ids,
                    },
                "dims": ["depth"],
                "op": "=", 
                "value": 1
            }
        ]
    }


def get_structure_search_kwargs(
    injection_structures=None, 
    primary_structure_only=True, 
    target_domain=None, 
    target_threshold=DEFAULT_TARGET_THRESHOLD, 
    transgenic_lines=None, 
    product_ids=None, 
    injection_domain=None, 
    injection_threshold=DEFAULT_TARGET_THRESHOLD,  # these are not actually used in the injection structures search
    showDetail=0, 
    startRow=0, 
    numRows='all'
    ):
    '''

    Notes
    -----
    http://stash.corp.alleninstitute.org/projects/INF/repos/mouse_conn_server/browse/HeatmapServer.cpp#1721

    '''

    filters = []

    if injection_structures is not None:
        filters.extend(build_injection_structures_clause(injection_structures, primary_structure_only))

    if target_domain is not None:
        hem, sids = decode_domain_str(target_domain)
    else:
        hem = 'bilateral'
        sids = [997]
    filters.extend(build_target_domain_clause(sids, hem, target_threshold))

    if transgenic_lines is not None:
        filters.extend(build_transgenic_lines_clause(transgenic_lines))

    if product_ids is not None:
        filters.extend(build_product_clause(product_ids))

    return {
        'fields': STRUCTURE_SEARCH_FIELDS_FULL if showDetail else STRUCTURE_SEARCH_FIELDS_DEFAULT, 
        'coords': {
            'hemisphere': hem,
            'normalized': False,
            'injection': False,
        }, 
        'select': {},
        'filters': {
            'and': [
                {
                'dims': 'structure',
                'any': {'and': filters}
                },
                nonoverlapping_structures_clause(sids)
            ]
        }
    }


def postprocess_search_differential_rows(df, showDetail):

    df['volume'] = df.apply(
        lambda row: np.nansum(row['volume']),
        axis=1
    )
    df = df.rename(columns={'volume': 'sum', 'data_set_id': 'id'})
    df['num-voxels'] = None

    if not showDetail:
        df = df.sort_values('sum', ascending=False)
        return df

    df = df.drop(columns=['transgenic_line_id'])
    df = postprocess_injection_coordinates(df)
    df = postprocess_injection_structures(df)

    df = df.sort_values('injection_volume', ascending=False)

    df = df.rename(columns={
        'injection_structures': 'injection-structures',
        'structure_name': 'structure-name',
        'injection_volume': 'injection-volume',
        'product_id': 'product-id',
        'structure_id': 'structure-id',
        'specimen_name': 'name',
        'structure_abbrev': 'structure-abbrev',
        'transgenic_line': 'transgenic-line',
        'structure_color': 'structure-color'
    })
    
    df['id'] = df['id'].astype(int)
    df['structure-id'] = df['structure-id'].astype(int)
    df['product-id'] = df['product-id'].astype(int)

    return df