from setuptools import setup

setup(
    name='NumericalAnalysis',
    version='0.1',
    packages=['numan'],
    author='mstrand',
    author_email='mwstrand@uci.edu',
    description=open('README.rst').read(),
    install_requires=['numpy'],
    url='https://github.com/mwstrand/numerical-analysis'
)
