# -*- coding: utf-8 -*-

from .perlin import *
from .surface import SurfaceGrid
from .playground import (
    make_triangular_grid,
)

__all__ = [
    # From perlin.py
    'perlin',
    'extend2d',
    # From grid.py
    'SurfaceGrid',
    # From playground.py
    'make_triangular_grid',
]

try:
    from pkg_resources import get_distribution
    __version__ = get_distribution('pythonperlin').version
except Exception:
    __version__ = '0.1.0'