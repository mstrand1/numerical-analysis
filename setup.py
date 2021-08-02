import numpy as np
from distutils.core import setup

setup(
    name='NumericalAnalysis',
    version='0.1',
    packages=['numan'],
    author='mstrand',
    author_email='mwstrand@uci.edu',
    description=open('README.md').read(),
    install_requires=['numpy']
)
