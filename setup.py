from io import open
from setuptools import find_packages, setup

setup(
    name="SMURF",
    version='0.0.0',
    author="Joshua Feinglass",
    author_email="joshua.feinglass@asu.edu",
    description="PyTorch implementation of SMURF",
    long_description=open("README.md", "r", encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    keywords='RoBERTa BERT NLP deep learning google metric image captioning video',
    license='MIT',
    url="https://github.com/JoshuaFeinglass/SMURF",

    install_requires=['torch>=1.0.0',
                      'numpy',
                      'pandas>=1.0.1',
                      'matplotlib',
                      'transformers>=3.0.0'
                      'nltk'
                      ],
    include_package_data=True,
    python_requires='>=3.6',
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],

)
