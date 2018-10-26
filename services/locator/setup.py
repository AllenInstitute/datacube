from setuptools import setup, find_packages

setup(
    name = 'locator',
    version = '0.1.0',
    description = "",
    author = "",
    author_email = "",
    url = '',
    packages = find_packages(),
    include_package_data=True,
    install_requires=[
        "six",
        "cachetools",
        "numpy",
        "Twisted",
        "requests",
        "txaio",
        "h5py",
        "scipy",
        "pynrrd",
        "Pillow",
        "autobahn",
        "lz4",
    ],
    extras_require={
        'dev': [
        ],
        'test': [
            "mock",
            "pytest",
        ],
    },
    setup_requires=[],
    python_requires=">=3.6",
)

