#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'Click>=7.0',
]

test_requirements = []

setup(
    author="Mahdy Shirdel",
    author_email='mshirdel@utexas.edu',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="this is the repo for the uq of 4th kind paper. ",
    entry_points={
        'console_scripts': [
            'uq4k=uq4k.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='uq4k',
    name='uq4k',
    packages=find_packages(include=['uq4k', 'uq4k.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/uq4k/uq4k',
    version='0.1.0',
    zip_safe=False,
)
