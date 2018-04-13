import pytest
import locator.classes.filmstrip_locator as fsl
import locator.configuration_manager as cm
import numpy as np
import os

@pytest.fixture()
def context_manager():
    env_vars = os.path.join(os.path.dirname(__file__),
                            '..',
                            'env_vars.json')
    return cm.ConfigurationManager(env_vars)

@pytest.fixture()
def filmstrip(context_manager):
    return fsl.FilmStripLocator(context_manager)

def test_get_rotation_matrix(filmstrip):
    c1 = np.array([1,1,1,1])

    mat = np.array(filmstrip.get_rotation_matrix(0, "horizontal"))
    c2 = np.dot(mat, c1)
    assert np.allclose(c2, [1,1,1,1])

    mat = np.array(filmstrip.get_rotation_matrix(np.pi, "horizontal"))
    c2 = np.dot(mat, c1)
    assert np.allclose(c2, [-1,1,-1,1])

    mat = np.array(filmstrip.get_rotation_matrix(0, "vertical"))
    c2 = np.dot(mat, c1)
    assert np.allclose(c2, [1,1,1,1])

    mat = np.array(filmstrip.get_rotation_matrix(np.pi, "vertical"))
    c2 = np.dot(mat, c1)
    assert np.allclose(c2, [1,-1,-1,1])
    
    
    
