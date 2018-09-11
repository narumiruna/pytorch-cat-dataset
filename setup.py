from setuptools import setup, find_packages

install_requires = [
    'pillow'
    'torch',
    'torchvision',
    'numpy',
]

setup(
    name='cat_dataset',
    version='0.1',
    author="Narumi",
    author_email="weaper@gmail.com",
    packages=find_packages(),
    install_requires=install_requires)
