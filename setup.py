from setuptools import setup, find_packages
# pip install -e .
setup(
    name='tsvreader',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},  # remove src prefix
    entry_points={
        'console_scripts': [
            #terminal command = module:function
            'tsvreader = TSVReader:main', #tsvreader <args>
        ],
    },
    python_requires='>=3.6',
)

