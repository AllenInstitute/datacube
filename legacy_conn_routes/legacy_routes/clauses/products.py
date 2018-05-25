import six

from legacy_routes.clauses.base import id_containment_filter


def build_product_clause(product_ids='all'):
    ''' 
    '''

    return id_containment_filter('product_id', product_ids)