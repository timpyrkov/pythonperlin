import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="pythonperlin",
    version="0.0.1",
    author="Tim Pyrkov",
    author_email="tim.pyrkov@gmail.com",
    description="Perlin noise in python - seamlessly tile in any dimensions",
    long_description=read("README.md"),
    license = "MIT License",
    long_description_content_type="text/markdown",
    url="https://github.com/timpyrkov/pythonperlin",
    packages=find_packages(exclude=("docs")),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.9",
    include_package_data=True,
    install_requires=[
        "numpy", 
    ],
)

