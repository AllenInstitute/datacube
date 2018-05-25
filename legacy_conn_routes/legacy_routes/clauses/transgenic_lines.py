import six

from legacy_routes.clauses.base import id_containment_filter


def build_transgenic_lines_clause(transgenic_line_ids='all'):
    '''
    '''

    return id_containment_filter('transgenic_line_id', transgenic_line_ids)