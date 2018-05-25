import six

def id_containment_filter(field, ids='all'):
    '''
    '''

    if isinstance(ids, six.string_types) and ids == 'all':
        return []

    if isinstance(ids, six.integer_types):
        ids = [ids]

    return [{
        'field': field,
        'op': 'in',
        'value': ids
    }]