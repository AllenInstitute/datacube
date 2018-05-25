import pytest
import numpy as np

import legacy_routes.clauses.base as base


def test_id_containment_filter():

    obt = base.id_containment_filter('my_field', ids=[1, 2, 3])[0]

    assert( obt['field'] == 'my_field' )
    assert(np.allclose( obt['value'], [1, 2, 3] ))