# Copyright (c) 2017 Salas Lin (leVirve)
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from setuptools import setup


def version():
    with open('onegan/__init__.py') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.replace("'", '').split()[-1]


setup(
    name='onegan',
    version=version(),
    url='http://github.com/leVirve/OneGAN',
    description='One GAN framewrok for fast development setups.',
    author='Salas Lin (leVirve)',
    author_email='gae.m.project@gmail.com',
    license='MIT',
    platforms='any',
    packages=['onegan'],
    zip_safe=False,
    keywords='GAN framework',
    install_requires=[
        'tensorboardX'
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
