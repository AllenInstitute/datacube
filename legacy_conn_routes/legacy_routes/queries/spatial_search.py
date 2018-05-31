
from legacy_routes.clauses.injection_structures import build_injection_structures_clause
from legacy_routes.clauses.products import build_product_clause
from legacy_routes.clauses.transgenic_lines import build_transgenic_lines_clause
from legacy_routes.utilities.response import postprocess_injection_coordinates, postprocess_injection_structures


SPATIAL_SEARCH_DETAILED_FIELDS = [
    "data_set_id",
    "experiment",
    "transgenic_line",
    "product_id",
    "structure_id",
    "structure_abbrev",
    "structure_name",
    "primary_structure_color",
    "specimen_name",
    "injection_volume",
    "injection_structures",
    "injection_x",
    "injection_y",
    "injection_z",
    "gender",
    "strain",
    "projection"
]


def get_spatial_search_kwargs(
    transgenic_lines=None,
    seedPoint=None,
    dataset=None,
    injection_structures=None,
    primary_structure_only=None,
    product_ids=None,
    showDetail=False, 
    sortOrder=None, 
    startRow=None, 
    numRows='all'
):

    filters = []

    if product_ids is not None:
        filters.extend(build_product_clause(product_ids))

    if transgenic_lines is not None:
        filters.extend(build_transgenic_lines_clause(transgenic_lines))

    if injection_structures is not None:
        filters.extend(build_injection_structures_clause(injection_structures, primary_structure_only))

    return {
        'fields': SPATIAL_SEARCH_DETAILED_FIELDS,
        'filters': filters,
        'coords': {
        'anterior_posterior': seedPoint[0],
        'superior_inferior': seedPoint[1],
        'left_right': seedPoint[2]
        },
        'select': {}
    }


def postprocess_spatial_search(df, ccf_store=None):

    df['num-voxels'] = None
    df = postprocess_injection_coordinates(df)    

    df['streamline'] = df.apply(lambda row: update_streamline(row['streamline'], row['injection-coordinates']), axis=1)
    df['density'] = df.apply(lambda row: row['streamline']['density'], axis=1)
    df['path'] = df.apply( lambda row: [ point for point in row['streamline']['coords'] ], axis=1)

    df = postprocess_injection_structures(df, ccf_store)

    df = df.rename(columns={
        'data_set_id': 'id',
        'transgenic_line': 'transgenic-line',
        'injection_structures': 'injection-structures',
        'injection_volume': 'injection-volume',
        'structure_name': 'structure-name',
        'structure_abbrev': 'structure-abbrev',
        'primary_structure_color': 'structure-color',
        'specimen_name': 'name',
        'product_id': 'product-id',
        'structure_id': 'structure-id',
    })

    df = df.drop(columns=['streamline'])
    return df


def update_streamline(streamline, injection_coordinates):
    streamline['coords'].append({
        'coord': injection_coordinates,
        'density': 0,
        'intensity': 0
    })

    return streamline