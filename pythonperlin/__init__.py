# -*- coding: utf-8 -*-

from .perlin import *
from .worley import *
from .surface import *

# __all__ = [
#     # From perlin.py
#     'perlin',
#     'extend2d',
#     'SurfaceGrid',
#     'show_3D',
# ]

try:
    from pkg_resources import get_distribution
    __version__ = get_distribution('pythonperlin').version
except Exception:
    __version__ = '0.1.0'