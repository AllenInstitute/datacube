
from setuptools import setup, find_packages

setup(
    name = 'legacy_routes',
    version = '0.1.0',
    description = """Translates mouseconn-style http requests to datacube ws requests (and responds)""",
    author = "nileg",
    author_email = "nileg@alleninstitute.org",
    url = 'http://nileg@stash.corp.alleninstitute.org/scm/~nileg/legacy_routes.git',
    packages = find_packages(),
    include_package_data=True,
    install_requires=[
        "twisted>=18.4.0",
        "autobahn>=18.4.1",
        "klein>=17.10.0",
        "simplejson>=3.14.0",
        "pyopenssl>=17.5.0",
        "numpy>=1.14.3",
        "pandas>=0.22.0",
    ],
    extras_require={
        'test': [
            "pytest>=2.9.2",
            "coverage>=3.7.1",
            "pytest-cov>=2.2.1",
            "requests>=2.18.4",
            "pylint>=1.8.4",
        ]
    },
    setup_requires=['pytest-runner'],
    python_requires=">=3.6",
)

