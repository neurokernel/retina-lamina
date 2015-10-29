#!/usr/bin/env python

import setuptools
from distutils.core import setup

setup(name='neurokernel-retlam',
      version='1.0',
      install_requires=[
        'configobj >= 5.0.0',
        'neurokernel >= 0.1',
        'neurokernel-retina >= 1.0',
        'neurokernel-lamina >= 1.0'
      ]
     )
