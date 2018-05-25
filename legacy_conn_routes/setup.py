
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
    setup_requires=['pytest-runner'],
) 