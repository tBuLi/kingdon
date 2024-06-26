#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst', encoding='utf-8') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['numpy', 'sympy', 'anywidget']

test_requirements = ['pytest>=3', ]

setup(
    author="Martin Roelfs",
    author_email='martinroelfs@yahoo.com',
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    description="Pythonic Geometric Algebra Package",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='kingdon',
    name='kingdon',
    packages=find_packages(include=['kingdon', 'kingdon.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/tbuli/kingdon',
    version='1.0.5',
    zip_safe=False,
)
