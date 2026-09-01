#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst', encoding='utf-8') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['numpy', 'sympy', 'anywidget']

# The array tests are parametrized over the array libraries that einops and array-api-compat
# both support. Those libraries are not dependencies of kingdon; they are merely tested with to ensure compatibility.
# The CI installs them by hand:
#   pip install torch --index-url https://download.pytorch.org/whl/cpu
#   pip install jax
#   pip install "cupy-cuda12x[ctk]"  # needs a GPU; match 12x/13x to your driver's CUDA version.
# The [ctk] extra is what makes cupy usable: it installs the CUDA libraries as wheels, which
# cuda-pathfinder prefers over a system-wide CUDA toolkit. Without it cupy picks up whichever
# toolkit is installed, and a version mismatch there is an access violation rather than an error.
test_requirements = ['pytest>=3', 'einops', 'array-api-compat']

setup(
    author="Martin Roelfs",
    author_email='martinroelfs@yahoo.com',
    # numpy, which kingdon cannot do without, requires 3.11 itself.
    python_requires='>=3.11',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
    ],
    description="Pythonic Geometric Algebra Package",
    install_requires=requirements,
    extras_require={'test': test_requirements},
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='kingdon',
    name='kingdon',
    packages=find_packages(include=['kingdon', 'kingdon.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/tbuli/kingdon',
    version='2.1.1',
    zip_safe=False,
)
