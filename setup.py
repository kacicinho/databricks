from setuptools import find_packages, setup
from databricks_jobs import __version__

setup(
    name='databricks_jobs',
    packages=find_packages(exclude=['tests', 'tests.*']),
    setup_requires=['wheel'],
    version=__version__,
    description='Repository containing the main script for the Data team jobs',
    author=''
)
