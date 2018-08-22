# -*- coding=utf-8 -*-

from distutils.core import setup
from setuptools import find_packages

setup(
    name='inkslab',
    version='0.1',
    description='Chinese nlp platform',
    long_description=open("README.md", encoding='u8').read(),
    license="Apache License 2.0",
    keywords='DeepLearning NLP Tensorflow',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    install_requires=['numpy', 'tensorflow'],
)
