from setuptools import setup
import datacubesdk

setup(name='datacubesdk',
    version=datacubesdk.__version__,
    description='Datacube tools and scripts',
    url='https://github.com/AllenInstitute/DatacubeSDK.git',
    author='Chris Barber',
    author_email='chrisba@alleninstitute.org',
    packages=['datacubesdk'],
    install_requires=[
        'allensdk'
    ])
