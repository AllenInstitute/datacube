import sys, os
path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, path + '/../../')

import pytest
import numpy as np
from mock import patch, mock_open, MagicMock, PropertyMock
import cachetools
from PIL import Image
import base64
import io

import locator.classes.stack_volume_slice as svs
import locator.classes.metaimage as mi
import locator.configuration_manager as cm

@pytest.fixture()
def context_manager():
    env_vars = os.path.join(os.path.dirname(__file__),
                            '..',
                            'env_vars.json')
    return cm.ConfigurationManager(env_vars)
    
@pytest.fixture()
def stack_volume_slice(context_manager):
    return svs.StackVolumeSlice(context_manager)

@pytest.fixture()
def stack_volume_image_info(context_manager):
    return svs.StackVolumeImageInfo(context_manager)

def test_16b_to_8b():
    im16 = np.random.random((10,10,3))*1000.0
    vr = [ 0, 100,
           0, 100, 
           0, 100 ]

    im8 = svs.image_16b_to_8b(im16, vr)
    assert im8.min() >= 0
    assert im8.max() <= 255

def test_image_jpeg_response():
    im = np.random.randint(low=0, high=255, size=(10,10,3)).astype(np.uint8)
    res = svs.image_jpeg_response(im, 40)
    data = res['data']
    validate_jpeg_data(data, 10, 10)

def test_plane_width_height_index():
    plane = 'coronal'
    width = None
    height = 100
    index = 2000
    dims = [100,100,100]
    spacing = [10., 10., 100.]
    max_dimension = 2048

    width, height, index, dims = svs.plane_width_height_index(plane, 
                                                              width, height, index, 
                                                              dims, spacing, max_dimension)

    assert width == 100
    assert height == 100
    assert index == 20
    assert np.allclose(dims, svs.PLANE_DIMENSION_ORDER[plane])

def test_get_slice(stack_volume_slice):
    plane = 'coronal'
    width = None
    height = 100
    index = 500
    dims = [100,100,10]
    spacing = [10., 10., 100.]
    sdir = '/made/up/path'
    vr = [ 0, 100,
           0, 100, 
           0, 100 ]

    with patch.object(svs.StackVolumeSlice, 'image_metadata',
                      return_value={ 'DimSize': dims, 'ElementSpacing': spacing }):
        with patch.object(cachetools.LRUCache, '__getitem__',
                          return_value=(np.random.random(dims),
                                        np.random.random(dims),
                                        np.random.random(dims))):
            resp = stack_volume_slice.get(sdir, plane, index, width, height, vr)

            data = resp['data']
            
            validate_jpeg_data(data, 100, height)

def validate_jpeg_data(data, width, height):
    jpeg_prefix = "data:image/jpeg;base64"
    assert data.startswith(jpeg_prefix)
    img_bytes = base64.b64decode(data[len(jpeg_prefix):])
    img = Image.open(io.BytesIO(img_bytes))
    assert img.size == (width, height)
    
def test_get_image_info(stack_volume_image_info):
    sdir = '/made/up/path'
    dim_size = [10,10,10]
    spacing = [10., 10., 10.]

    with patch.object(mi.MetaImage, 'metadata', PropertyMock(return_value={ 'DimSize': [10,10,10], 
                                                                            'ElementSpacing': spacing })):
    
        resp = stack_volume_image_info.get(sdir)
        assert np.allclose(resp['data']['spacing'], spacing)
        assert np.allclose(resp['data']['size'], dim_size)
    

        
        
    
    


    
