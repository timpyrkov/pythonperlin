#!/usr/bin/env python
# -*- coding: utf8 -*-

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict, Any, Literal, Union
from .perlin import _shape_and_dens_to_tuples


def _tilt_axis(x, tilt: Literal["forward", "backward", "inward", "outward"] = 'forward'):
    if x.size == 0:
        return x
    if tilt in ['forward', 'inward']:
        x[::2] += 0.5
    if tilt in ['backward', 'outward']:
        x[1::2] += 0.5
    if tilt == 'inward':
        x[::2, -1] = np.nan
    if tilt == 'outward':
        x[1::2, -1] = np.nan
    return x


def _reverse_tilt(tilt: Literal["forward", "backward", "inward", "outward"]):
    if tilt == "forward":
        tilt = "backward"
    elif tilt == "backward":
        tilt = "forward"
    elif tilt == "inward":
        tilt = "outward"
    elif tilt == "outward":
        tilt = "inward"
    return tilt



def make_triangular_grid(
    shape: Union[int, Tuple[int, ...]],
    dens: Union[int, Tuple[int, ...]] = 1,
    center: Optional[Tuple[int, int]] = None,
    stride: float = 1.0,
    padding: bool = False,
    base: Literal["horizontal", "vertical"] = 'horizontal',
    tilt: Literal["forward", "backward", "inward", "outward"] = 'inward',
) -> np.ndarray:
    """
    Make a triangular grid.

    Parameters
    ----------
    shape : Union[int, Tuple[int, ...]]
        The shape of the grid.
    dens : Union[int, Tuple[int, ...]], default 1
        The density of the grid.
    center : Optional[Tuple[int, int]], default None
        The center of the grid.
    stride : float, default 1.0
        The stride of the grid.
    padding : bool, default False
        If True, pad the grid with one row and one column in each direction.
    base : Literal["horizontal", "vertical"], default "horizontal"
        The base of the grid.
    tilt : Literal["forward", "backward", "inward", "outward"], default "inward"
        The tilt of the grid.
    """
    # Convert shape and dens to tuples
    shape, dens = _shape_and_dens_to_tuples(shape, dens=dens)
    if len(shape) != len(dens) or len(shape) != 2:
        raise ValueError("shape and dens must have the same number of elements and must be 2D")
    
    # Calculate grid center and bounding box
    size = tuple(s * d for s, d in zip(shape, dens))
    x0 = 0.5 * size[0] * stride
    y0 = 0.5 * size[1] * stride
    if base == 'horizontal':
        y0 *= np.sqrt(3) / 2
        if tilt in ['forward', 'backward']:
            x0 += 0.25 * stride
    elif base == 'vertical':
        x0 *= np.sqrt(3) / 2
        if tilt in ['forward', 'backward']:
            y0 += 0.25 * stride

    # Assign center if not provided
    if center is None:
        center = (x0, y0)

    # Assign bounding box
    box = ((center[0] - x0, center[1] - y0), (center[0] + x0, center[1] + y0))
    
    # Calculate the size of the grid
    size = tuple((s + 2 * int(padding)) * d for s, d in zip(shape, dens))

    # Create the grid
    x, y = [np.linspace(0, s, s + 1) for s in size]
    x, y = np.meshgrid(x, y)

    # Pad the grid
    if padding:
        x -= dens[0]
        y -= dens[1]
        if base == 'horizontal' and dens[1] % 2 == 1:
            tilt = _reverse_tilt(tilt)
        if base == 'vertical' and dens[0] % 2 == 1:
            tilt = _reverse_tilt(tilt)

    if base == 'horizontal':
        y *= np.sqrt(3) / 2
        x = _tilt_axis(x, tilt)
        y[np.isnan(x)] = np.nan
    elif base == 'vertical':
        x *= np.sqrt(3) / 2
        y = _tilt_axis(y.T, tilt).T
        x[np.isnan(y)] = np.nan

    # Scale and shift the grid
    x = x * stride + center[0]
    y = y * stride + center[1]

    # Create indices of the grid faces
    face_idxs = []
    for i in range(x.shape[0] - 1):
        for j in range(x.shape[1] - 1):
            level = i + int(tilt in ['backward', 'outward'])
            if base == 'vertical':
                level = j + int(tilt in ['backward', 'outward'])
            if level % 2 == 0:
                if np.isfinite(x[i+1, j+1]):
                    face_idxs.append([(i, j), (i+1, j), (i+1, j+1)])
                    if np.isfinite(x[i, j+1]):
                        face_idxs.append([(i, j), (i+1, j+1), (i, j+1)])
            else:
                if np.isfinite(x[i, j+1]):
                    face_idxs.append([(i, j), (i+1, j), (i, j+1)])
                    if np.isfinite(x[i+1, j+1]):
                        face_idxs.append([(i+1, j), (i+1, j+1), (i, j+1)])

    # Create the coordinates of the grid faces
    face_coords = []
    for i in face_idxs:
        face_coords.append([(x[i[0][0], i[0][1]], y[i[0][0], i[0][1]]),
                            (x[i[1][0], i[1][1]], y[i[1][0], i[1][1]]),
                            (x[i[2][0], i[2][1]], y[i[2][0], i[2][1]])])
    return face_coords


def find_triangular_grid_shape(
    size: Tuple[int, int],
    nrows: Optional[int] = None,
    ncols: Optional[int] = None,
    base: Literal["horizontal", "vertical"] = 'horizontal',
) -> Tuple[int, int]:
    """
    Find triangular grid shapes that best fit the canvas size.
    Provide nrows to find ncols, or provide ncols to find nrows.

    Parameters
    ----------
    size : Tuple[int, int]
        The size of the canvas.
    nrows : int, default None
        The number of rows in the grid.
    ncols : int, default None
        The number of columns in the grid.
    base : Literal["horizontal", "vertical"], default "horizontal"
        The base of the grid.

    Raises
    ------
    ValueError
        If the canvas size is too small or if both nrows and ncols are provided.
    """
    width, height = size

    if width < 15 or height < 15:
        raise ValueError("Canvas size too small, must be at least 15x15")
    
    if nrows is None and ncols is None:
        raise ValueError("Either nrows or ncols must be provided")
    
    if nrows is not None and ncols is not None:
        raise ValueError("Either nrows or ncols must be provided, not both")
    
    if base == 'horizontal':
        if ncols is not None:
            dx = width / ncols
            dy = dx * np.sqrt(3) / 2
            nrows = int(height / dy)
        ny = 2 * int(nrows / 2)
        dy = int(height / ny)
        dx = int(dy * 2 / np.sqrt(3))
        nx = int(width / dx)

        for i in range(1,4):
            for j in range(1,4):
                if nx % i == 0 and ny % j == 0:
                    shape = f'shape=({nx // i}, {ny // j})'
                    dens = f'dens=({i}, {j})'
                    stride = f'stride={dx}'
                    center = f'center=({width // 2}, {height // 2})'
                    print(f'{shape}, {dens}, {stride}, {center}, base="{base}"')
        
    elif base == 'vertical':
        if nrows is not None:
            dy = height / nrows
            dx = dy * np.sqrt(3) / 2
            ncols = int(width / dx)
        nx = 2 * int(ncols / 2)
        dx = int(width / nx)
        dy = int(dx * 2 / np.sqrt(3))
        ny = int(height / dy)

        for i in range(1,4):
            for j in range(1,4):
                if nx % i == 0 and ny % j == 0:
                    shape = f'shape=({nx // i}, {ny // j})'
                    dens = f'dens=({i}, {j})'
                    stride = f'stride={dy}'
                    center = f'center=({width // 2}, {height // 2})'
                    print(f'{shape}, {dens}, {stride}, {center}, base="{base}"')

    return












def make_hexagonal_grid(
    shape: Union[int, Tuple[int, ...]],
    dens: Union[int, Tuple[int, ...]] = 1,
    center: Optional[Tuple[int, int]] = None,
    stride: float = 1.0,
    padding: bool = False,
    base: Literal["horizontal", "vertical"] = 'horizontal',
    tilt: Literal["forward", "backward", "inward", "outward"] = 'inward',
) -> np.ndarray:
    """
    Make a hexagonal grid.

    Parameters
    ----------
    shape : Union[int, Tuple[int, ...]]
        The shape of the grid.
    dens : Union[int, Tuple[int, ...]], default 1
        The density of the grid.
    center : Optional[Tuple[int, int]], default None
        The center of the grid.
    stride : float, default 1.0
        The stride of the grid.
    padding : bool, default False
        If True, pad the grid with one row and one column in each direction.
    base : Literal["horizontal", "vertical"], default "horizontal"
        The base of the grid.
    tilt : Literal["forward", "backward", "inward", "outward"], default "inward"
        The tilt of the grid.
    """
    # Convert shape and dens to tuples
    shape, dens = _shape_and_dens_to_tuples(shape, dens=dens)
    if len(shape) != len(dens) or len(shape) != 2:
        raise ValueError("shape and dens must have the same number of elements and must be 2D")
    
    # Calculate the size of the grid
    size = tuple((s + 2 * int(padding)) * d for s, d in zip(shape, dens))

    # Create the grid
    x, y = [np.linspace(0, s, s + 1) for s in size]
    x, y = np.meshgrid(x, y)

    # Pad the grid
    if padding:
        x -= dens[0]
        y -= dens[1]
        if base == 'horizontal' and dens[1] % 2 == 1:
            tilt = _reverse_tilt(tilt)
        if base == 'vertical' and dens[0] % 2 == 1:
            tilt = _reverse_tilt(tilt)


    if base == 'horizontal':
        y *= np.sqrt(3) / 2
        x = _tilt_axis(x, tilt)
        y[np.isnan(x)] = np.nan
    elif base == 'vertical':
        x *= np.sqrt(3) / 2
        y = _tilt_axis(y.T, tilt).T
        x[np.isnan(y)] = np.nan

    # Scale and shift the grid
    x *= stride
    y *= stride
    if center is not None:
        x += center[0]
        y += center[1]

    # Create indices of the grid faces
    face_idxs = []
    for i in range(x.shape[0] - 1):
        for j in range(x.shape[1] - 1):
            level = i + int(tilt in ['backward', 'outward'])
            if base == 'vertical':
                level = j + int(tilt in ['backward', 'outward'])
            if level % 2 == 0:
                if np.isfinite(x[i+1, j+1]):
                    face_idxs.append([(i, j), (i+1, j), (i+1, j+1)])
                    if np.isfinite(x[i, j+1]):
                        face_idxs.append([(i, j), (i+1, j+1), (i, j+1)])
            else:
                if np.isfinite(x[i, j+1]):
                    face_idxs.append([(i, j), (i+1, j), (i, j+1)])
                    if np.isfinite(x[i+1, j+1]):
                        face_idxs.append([(i+1, j), (i+1, j+1), (i, j+1)])

    # Create the coordinates of the grid faces
    face_coords = []
    for i in face_idxs:
        face_coords.append([(x[i[0][0], i[0][1]], y[i[0][0], i[0][1]]),
                            (x[i[1][0], i[1][1]], y[i[1][0], i[1][1]]),
                            (x[i[2][0], i[2][1]], y[i[2][0], i[2][1]])])
    return face_coords


