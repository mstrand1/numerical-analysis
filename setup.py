from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='numan',
    version='0.1',
    packages=find_packages(where='src'),
    author='mstrand',
    python_requires='>=3.6, <4',
    author_email='mwstrand@uci.edu',
    long_description='Numerical Analysis Algorithms from class.',
    long_description_content_type='text/markdown',
    install_requires=['numpy'],
    url='https://github.com/mwstrand/numerical-analysis'
)
