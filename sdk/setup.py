from setuptools import setup, find_packages
import datacubesdk

setup(name='datacubesdk',
    version=datacubesdk.__version__,
    description='Datacube tools and scripts',
    url='https://github.com/AllenInstitute/DatacubeSDK.git',
    author='Chris Barber',
    author_email='chrisba@alleninstitute.org',
    packages=find_packages(),
    install_requires=[
        'allensdk'
    ])
