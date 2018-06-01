import ast
import sys

HEMISPHERE_MAP = {
    'l': 'left',
    'r': 'right',
    'b': 'bilateral',
    '': 'bilateral'
}

DEFAULT_DOMAIN_THRESHOLD = ( 0.0, sys.float_info.max, )


def build_domain_clause(structure_ids, hemisphere='bilateral', injection=False, threshold=DEFAULT_DOMAIN_THRESHOLD):
    '''
    '''

    filters = []

    filters.append({
        'field': 'volume',
        'coords': {
            'injection': injection,
            'hemisphere': hemisphere,
            'normalized': False,
            'structure': structure_ids
        },
        'op': '>=',
        'value': threshold[0]
    })

    filters.append({
        'field': 'volume',
        'coords': {
            'injection': injection,
            'hemisphere': hemisphere,
            'normalized': False,
            'structure': structure_ids
        },
        'op': '<=',
        'value': threshold[1]
    })

    return [{
        'dims': 'structure',
        'any': {'and': filters}
    }]


def decode_domain_str(
    domain_str, 
    sep=':', 
    structure_sep=',',
    root_structure=997, 
    bilateral='bilateral', 
    acronym_id_map=None
    ):
    '''

    Notes
    -----
    supported formats are:
    1. {hemisphere}:{structure_id},{structure_acronym},... in whatever order
    2. {structure_id},{structure_acronym},... in whatever order
    4. nothing at all

    '''

    if isinstance(domain_str, str) and domain_str == '':
        return bilateral, [root_structure]

    if isinstance(domain_str, str) and domain_str.lower() in HEMISPHERE_MAP and not sep in domain_str:
        return HEMISPHERE_MAP[domain_str], [root_structure]

    if isinstance(domain_str, int):
        return bilateral, [domain_str]

    if not isinstance(domain_str, str):
        domain_str = structure_sep.join(map(str, domain_str))

    if sep in domain_str:
        hemisphere, domain_str = domain_str.split(sep)
        hemisphere = HEMISPHERE_MAP[hemisphere.lower()]

    else:
        hemisphere = bilateral

    if len(domain_str) == 0:
        domain_str = str(root_structure)

    structures = []
    for st in domain_str.split(structure_sep):
        try:
            structures.append(int(st))
        except(ValueError, TypeError):
            if acronym_id_map is None:
                raise TypeError('unable to parse structure {} without an acronym_id_map'.format(st))
            structures.append(acronym_id_map[st])

    return hemisphere, structures