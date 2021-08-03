from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='nanalysis',
    version='1.0',
    packages=find_packages(where='src'),
    author='mstrand',
    author_email='mwstrand@uci.edu',
    long_description='Numerical Analysis Algorithms from class.',
    long_description_content_type='text/markdown',
    install_requires=['numpy'],
    setup_requires=['numpy'],
    url='https://github.com/mwstrand/numerical-analysis'
)
