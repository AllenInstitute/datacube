
import numpy as np

from legacy_routes.clauses.domain import decode_domain_str
from legacy_routes.clauses.injection_structures import build_injection_structures_clause
from legacy_routes.clauses.products import build_product_clause
from legacy_routes.utilities.response import postprocess_injection_coordinates, postprocess_injection_structures
from legacy_routes.clauses.transgenic_lines import build_transgenic_lines_clause


CORR_SEARCH_DETAILED_FIELDS = [
    "data_set_id",
    "experiment",
    "transgenic_line",
    "product_id",
    "structure_id",
    "structure_abbrev",
    "structure_color", 
    "structure_name",
    "specimen_name",
    "injection_volume",
    "injection_structures",
    "injection_x",
    "injection_y",
    "injection_z",
    "gender",
    "strain"    
]


CORR_SEARCH_BASE_FIELDS = [
    'data_set_id',
]


def get_correlation_search_kwargs(
    product_ids=None, transgenic_lines=None, 
    domain=None, 
    injection_structures=None, primary_structure_only=None, 
    showDetail=False, seed=None,
    sortOrder=None, startRow=None, numRows='all'
):

    filters = []
    experiment_filters = []
    
    if domain is not None:
        hem, sids = decode_domain_str(domain, sep=':')
        filters.append({
            'field': 'ccf_structures_flat',
            'op': 'in',
            'value': sids
        })

    if product_ids is not None:
        experiment_filters.extend(build_product_clause(product_ids))

    if transgenic_lines is not None:
        experiment_filters.extend(build_transgenic_lines_clause(transgenic_lines))

    if injection_structures is not None:
        experiment_filters.extend(build_injection_structures_clause(injection_structures, primary_structure_only))

    if experiment_filters:
        filters.append({
            'field': 'experiment',
            'op': '>',
            'value': -1
        })

        filters = {"or": [{"field": 'experiment', "op": "=", "value": seed}, {'and': filters + experiment_filters}]}

    return {
        'fields': CORR_SEARCH_DETAILED_FIELDS if showDetail else CORR_SEARCH_BASE_FIELDS,
        'filters': filters
    }


def postprocess_correlation_search(df, showDetail=False):

    df = df.rename(columns={
        'corr': 'r',
        'data_set_id': 'id',
    })
    df['num-voxels'] = None # TODO: this can be dropped entirely, maybe?

    if not showDetail: 
        return df

    df = postprocess_injection_coordinates(df)
    df = postprocess_injection_structures(df)

    df = df.rename(columns={
        'transgenic_line': 'transgenic-line',
        'injection_structures': 'injection-structures',
        'injection_volume': 'injection-volume',
        'structure_name': 'structure-name',
        'structure_abbrev': 'structure-abbrev',
        'structure_color': 'structure-color',
        'specimen_name': 'name',
        'product_id': 'product-id',
        'structure_id': 'structure-id',
    })

    df['id'] = df['id'].astype(int)

    return df