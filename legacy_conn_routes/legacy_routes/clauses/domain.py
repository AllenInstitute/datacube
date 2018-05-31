import sys

HEMISPHERE_MAP = {
    'l': 'left',
    'r': 'right',
    'b': 'bilateral'
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
            'normalized': False
        },
        'op': '>=',
        'value': threshold[0]
    })

    filters.append({
        'field': 'volume',
        'coords': {
            'injection': injection,
            'hemisphere': hemisphere,
            'normalized': False
        },
        'op': '<=',
        'value': threshold[1]
    })

    return [{'and': filters}]


def decode_domain_str(domain_str, sep=None, root_structure=997):

    if sep is None and domain_str[0] in HEMISPHERE_MAP:
        hemisphere = HEMISPHERE_MAP[domain_str[0].lower()]
        domain_str = domain_str[1:]
    elif sep:
        hemisphere, domain_str = domain_str.split(sep)
        hemisphere = HEMISPHERE_MAP[hemisphere.lower()]
    else:
        hemisphere = 'bilateral'

    if len(domain_str) == 0:
        domain_str = str(root_structure)

    return hemisphere, [int(ii) for ii in domain_str.split(',')]
