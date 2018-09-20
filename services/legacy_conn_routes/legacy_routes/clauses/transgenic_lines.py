import six

from legacy_routes.clauses.base import id_containment_filter


def build_transgenic_lines_clause(transgenic_line_ids='all'):
    '''
    '''

    wt_filter = {'field': 'transgenic_line_id', 'op': 'isnan'}

    if isinstance(transgenic_line_ids, (str, int)) and transgenic_line_ids != 'all':
        transgenic_line_ids = [transgenic_line_ids]
    elif transgenic_line_ids == 'all':
        return []

    lines = set()
    line_ids = set()
    for tli in transgenic_line_ids:
        if isinstance(tli, str):
            lines.add(tli)
        elif isinstance(tli, int):
            line_ids.add(tli)
        else:
            raise TypeError('type: {} not undestood for data: {}'.format(type(tli), tli))

    include_wt = False
    if 0 in line_ids:
        include_wt = True
        line_ids.remove(0)
    if '' in lines:
        include_wt = True
        lines.remove('')

    filters = []

    if include_wt:
        filters.append(wt_filter)

    if len(line_ids) > 0:
        filters.extend(id_containment_filter('transgenic_line_id', list(line_ids)))
    
    if len(lines) > 0:
        filters.extend(id_containment_filter('transgenic_line', list(lines)))

    if len(filters) > 0:
        return [{'or': filters}]
    else:
        return []

