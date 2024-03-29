import six

INJECTION_STRUCTURES_NPV_THRESHOLD = 0.005


def build_injection_structures_clause(injection_structures, 
                                      primary_structure_only=False, 
                                      npv_threshold=INJECTION_STRUCTURES_NPV_THRESHOLD,
                                      acronym_id_map=None):
    '''
    '''

    if isinstance(injection_structures, (six.integer_types, str)):
        injection_structures = [injection_structures]

    cast_structures = []
    for item in injection_structures:
        if isinstance(item, str):
            if acronym_id_map is None:
                raise TypeError('cannot convert string {} without an acronym_id_map'.format(item))
            item = acronym_id_map[item]
        cast_structures.append(item)
    injection_structures = cast_structures

    filters = []

    filters.append({
        "field": "volume",
        "coords": {
            "injection": True,
            "hemisphere": 'bilateral',
            "normalized": True
        },
        "op": ">=",
        "value": npv_threshold
    })

    filters.append({
        'dims': 'depth',
        'any': {
            'field': 'structures',
            'op': 'in',
            'value': injection_structures
        }
    })

    if primary_structure_only:
        filters.append({
            'field': 'is_primary',
            'op': '=',
            'value': True
        })

    return [{'and': filters}]


def build_injection_structure_acronym_clause(acronym):
    '''

    Notes
    -----
    For some reason, the mouseconn service calls this a structure NAME query

    see: 
    http://stash.corp.alleninstitute.org/projects/INF/repos/mouse_conn_server/browse/HeatmapDataSet.cpp#394
    http://stash.corp.alleninstitute.org/projects/INF/repos/mouse_conn_server/browse/HeatmapDataSet.cpp#564
    
    '''

    return [{
        'field': 'structureAbbrev',
        'op': '=',
        'value': acronym
    }]
