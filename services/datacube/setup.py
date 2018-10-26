from setuptools import setup, find_packages

setup(
    name = 'datacube-service',
    version = '0.1.0',
    description = "",
    author = "",
    author_email = "",
    url = '',
    packages = find_packages(),
    include_package_data=True,
    install_requires=[
        'autobahn',
        'Twisted',
        'numpy',
        'scipy',
        'txaio',
        'xarray',
        'zarr',
        'h5netcdf',
        'dask',
        'toolz',
        'txredisapi',
        'redis',
        'six',
        'Pillow',
        'lz4',
    ],
    extras_require={
        'dev': [
        ],
        'test': [
            'mock',
            'pytest',
            'pytest-redis',
        ],
    },
    setup_requires=[],
    python_requires=">=3.6",
)

