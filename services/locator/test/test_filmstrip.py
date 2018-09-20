import sys, os
path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, path + '/../../')

import pytest
import locator.classes.filmstrip_locator as fsl
import locator.configuration_manager as cm
import numpy as np

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

def test_rotate_volume_coordinate(filmstrip):

    # this thing is quite stateful, don't want the tests to mess it up
    old_cor = filmstrip.center_of_rotation
    
    frames_per_deg = filmstrip.frames / filmstrip.rotation

    filmstrip.center_of_rotation = np.array([1,0,1])
    c = filmstrip.rotate_volume_coordinate(np.array([0,0,0]), 0, "horizontal")
    assert np.allclose(c, [0,0,0])

    c = filmstrip.rotate_volume_coordinate(np.array([0,0,0]), frames_per_deg*90, "horizontal")
    assert np.allclose(c, [0,0,2])

    filmstrip.center_of_rotation = np.array([0,1,1])
    c = filmstrip.rotate_volume_coordinate(np.array([0,0,0]), 0, "vertical")
    assert np.allclose(c, [0,0,0])

    c = filmstrip.rotate_volume_coordinate(np.array([0,0,0]), frames_per_deg*180, "vertical")
    assert np.allclose(c, [0,2,2])

    # reset to old value
    filmstrip.center_of_rotation = old_cor
    
