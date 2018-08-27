from setuptools import setup, find_packages

setup(
    name = 'datacube',
    version = '0.1.0',
    description = "",
    author = "",
    author_email = "",
    url = '',
    packages = find_packages(),
    include_package_data=True,
    install_requires=[
        'crossbar>=18.4.1,!=18.5.*,!=18.6.*,!=18.7.1',
        'klein',
        'simplejson',
    ],
    extras_require={
        'dev': [
            'autobahn-python-repl',
            'restview',
            'yasha',
        ],
        'test': [
            'pytest',
        ]
    },
    setup_requires=[],
    python_requires=">=3.6",
)

