#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

import pathlib
from setuptools import find_packages, setup

README = pathlib.Path("docs/index.md").read_text()
HISTORY = pathlib.Path("CHANGELOG.md").read_text()

REQUIREMENTS = ["pandas", "numba"]

setup(
    author="Philipp Temminghoff",
    author_email="phil65@kodi.tv",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="Helper functions for Pandas DataFrames / Series",
    install_requires=REQUIREMENTS,
    license="MIT license",
    python_requires=">=3.6.0",
    long_description=README + "\n\n" + HISTORY,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="pandas",
    name="pandastools",
    packages=find_packages(),
    test_suite="tests",
    url="https://github.com/phil65/pandastools",
    version="0.5.0",
    zip_safe=False,
)
