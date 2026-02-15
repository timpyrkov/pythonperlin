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
    from importlib.metadata import version
    __version__ = version('pythonperlin')
except Exception:
    __version__ = '0.1.0'
