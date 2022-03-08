#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = ["cvxpy==1.1.18", "miniball==1.1.0", "optax==0.1.1", "numpy==1.22.2", "scipy==1.8.0"]

dev_requirments = [
    "ipykernel==6.9.1",
    "matplotlib==3.5.1",
    "flake8==3.7.8",
    "isort==5.9.1",
    "black==21.6b0",
    "pre-commit==2.13.0",
    "twine==1.14.0",
    "build",
]

setup(
    author="Mahdy Shirdel",
    author_email='mshirdel@utexas.edu',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="UQ4k: Uncertaininty Quantification of the 4th Kind",
    long_description_content_type="text/markdown",
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    include_package_data=True,
    keywords='uq4k',
    name='uq4k',
    packages=find_packages(include=['uq4k', 'uq4k.*']),
    test_suite='tests',
    extras_require={"dev": dev_requirments},
    url='https://github.com/uq4k/uq4k',
    version='0.1.0-beta',
    zip_safe=False,
)
