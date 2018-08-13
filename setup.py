# Copyright (c) 2017 Salas Lin (leVirve)
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from setuptools import setup, find_packages


def version():
    with open('onegan/__init__.py') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.replace("'", '').split()[-1]


def pip_requirements():
    return [
        'torchvision',
        'tensorboardX',
        'tqdm',
        'pyyaml',
    ]


setup(
    name='onegan',
    version=version(),
    url='http://github.com/leVirve/OneGAN',
    description='One GAN framework for fast development setups.',
    author='Salas Lin (leVirve)',
    author_email='gae.m.project@gmail.com',
    license='MIT',
    platforms='any',
    packages=find_packages(),
    zip_safe=False,
    keywords='GAN framework',
    install_requires=[
        *pip_requirements(),
        'numpy',
        'scipy',
        # 'opencv',
        'pillow >= 4.1.1',
        'torch'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Customer Service',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6'
    ]
)
