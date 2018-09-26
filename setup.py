from setuptools import setup

readme = open('README.rst').read()

setup(
    name="bootstrapped",
    version="0.0.1",
    description="Implementations of the percentile based bootstrap",
    author="Spencer Beecher",
    author_email="spencebeecher@gmail.com",
    packages=['bootstrapped'],
    long_description=readme,
    install_requires=[
        "matplotlib>=1.5.3",
        "numpy>=1.11.1",
        "pandas>=0.18.1",
    ],
    url='https://github.com/facebookincubator/bootstrapped',
)
