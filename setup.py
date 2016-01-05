#!/usr/bin/env python

import sys, os

# Install setuptools if it isn't available:
try:
    import setuptools
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()

from setuptools import find_packages
from setuptools import setup


NAME =               'neurokernel-retlam'
VERSION =            '1.0'
AUTHOR =             'Neurokernel Development Team'
AUTHOR_EMAIL =       'neurokernel-dev@columbia.edu'
URL =                'https://github.com/neurokernel/retina-lamina'
MAINTAINER =         AUTHOR
MAINTAINER_EMAIL =   AUTHOR_EMAIL
DESCRIPTION =        'Interface between the retina and lamina LPUs in Neurokernel'
LONG_DESCRIPTION =   DESCRIPTION
DOWNLOAD_URL =       URL
LICENSE =            'BSD'
CLASSIFIERS = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: BSD License',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Topic :: Scientific/Engineering',
    'Topic :: Software Development']

# Explicitly switch to parent directory of setup.py in case it
# is run from elsewhere:
os.chdir(os.path.dirname(os.path.realpath(__file__)))
PACKAGES =           find_packages()

if __name__ == "__main__":
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    setup(
        name = NAME,
        version = VERSION,
        author = AUTHOR,
        author_email = AUTHOR_EMAIL,
        license = LICENSE,
        classifiers = CLASSIFIERS,
        description = DESCRIPTION,
        long_description = LONG_DESCRIPTION,
        url = URL,
        maintainer = MAINTAINER,
        maintainer_email = MAINTAINER_EMAIL,
        packages = PACKAGES,
        include_package_data = True,
        install_requires=[
            'configobj >= 5.0.0',
            'neurokernel >= 0.1',
            'neurokernel-retina >= 1.0',
            'neurokernel-lamina >= 1.0'
        ],
        dependency_links=[
            'https://github.com/neurokernel/retina/tarball/master#egg=neurokernel-retina-1.0',
            'https://github.com/neurokernel/lamina/tarball/master#egg=neurokernel-lamina-1.0'
        ]
    )
