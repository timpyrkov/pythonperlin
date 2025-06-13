#!/usr/bin/env python
# -*- coding: utf8 -*-

import itertools
import numpy as np
from typing import Union, Tuple

def _shape_and_dens_to_tuples(
        *shape: Union[int, Tuple[int, ...]], 
        dens: Union[int, Tuple[int, ...]] = 1, 
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """
    Convert shape and dens to tuples. Check if the shape and dens are valid.

    Parameters
    ----------
    *shape : Union[int, Tuple[int, ...]]
        Shape of the noise grid.
    dens : Union[int, Tuple[int, ...]], default 1
        Number of interpolation points between grid nodes along each axis

    Returns
    -------
    shape : Tuple[int, ...]
        Shape as tuple
    dens : Tuple[int, ...]
        Dens as tuple
    """
    shape = tuple(shape[0]) if isinstance(shape[0], (tuple, list)) else shape
    if any(s < 1 for s in shape):
        raise ValueError("shape must be 1 or greater")
    if isinstance(dens, int):
        if dens < 1:
            raise ValueError("dens must be 1 or greater")
        dens = tuple([dens] * len(shape))
    if len(dens) != len(shape):
        raise ValueError("Shape and dens must have the same number of elements")
    if any(d < 1 for d in dens):
        raise ValueError("dens must be 1 or greater")
    return shape, dens


def _pad_array(arr):
    """
    Helper function to pad array by 2 layers in each direction with periodic (wrap) padding.

    Parameters
    ----------
    arr : ndarray
        Array to pad

    Returns
    -------
    arr_padded : ndarray
        Padded array
    """
    arr = np.asarray(arr)
    ndim = arr.ndim
    pad_width = []
    for i in range(ndim):
        pad_width.append((2, 2))
    arr_padded = np.pad(arr, pad_width, mode='wrap')
    # For axes with length 1, ensure the padding is correct (all values are the same)
    # np.pad with mode='wrap' already handles this by repeating the single value
    return arr_padded


def _make_noise(*shape, boundary=False, seed=None):
    """
    Helper function to generate grid of noise in range (0,1) in each grid cell.
    
    Parameters
    ----------
    *shape 
       Shape of the noise grid.
    boundary : bool, default False
        If True - pad with 2 layers of periodic boundary in each dimension
    seed : int, default None
        Numpy random seed

    Returns
    -------
    noise : ndarray
        Noise of shape = shape

    """
    shape = tuple(shape[0]) if isinstance(shape[0], (tuple, list)) else shape
    rng = np.random.default_rng(seed)
    noise = rng.uniform(0, 1, size=shape)
    if boundary:
        noise = _pad_array(noise)
    return noise


def _make_centers(*shape, boundary=False, seed=None):
    """
    Helper function to generate coordinate meshgrid and adds noise in range (0,1).
    
    Parameters
    ----------
    *shape
        Integers or tuple of integers
    boundary : bool, default False
        If True - pad with 1 layer of periodic boundary in each dimension
    seed : int, default None
        Numpy random seed

    Returns
    -------
    centers : ndarray
        Center coordinates of shape = (ndim, *shape), where ndim = len(shape)
    idx : ndarray
        Continuous integer indices of shape = (*shape,), running from 0 to N-1
    """
    shape = tuple(shape[0]) if isinstance(shape[0], (tuple, list)) else shape
    size = tuple([s + 4 for s in shape]) if boundary else shape
    centers = []
    ndim = len(shape)
    for i in range(ndim):
        sh = np.ones((ndim)).astype(int)
        sh[i] = size[i]
        x = np.arange(size[i]).reshape(sh)
        x = x - 2 if boundary else x
        for j in range(ndim):
            if j != i:
                x = np.repeat(x, size[j], axis=j)
        s = seed + i if seed is not None else None
        x = x + _make_noise(shape, boundary=boundary, seed=s)
        centers.append(x)
    centers = np.stack(centers)
    # Create continuous indices
    idx = np.arange(np.prod(shape)).reshape(shape)
    if boundary:
        idx = _pad_array(idx)
    return centers, idx


def _make_grid(*shape, dens=4):
    """
    Helper function to generate grid coordinates. 
    
    Parameters
    ----------
    *shape : Union[int, Tuple[int, ...]]
        Shape of the noise grid.
    dens : Union[int, Tuple[int, ...]], default 4
        Number of interpolation points between grid nodes along each axis

    Returns
    -------
    grid : ndarray
        Grid coordinates of shape = (ndim, *shape * dens), ndim = len(shape)

    """
    shape, dens = _shape_and_dens_to_tuples(*shape, dens=dens)
    size = tuple([s * d for (s, d) in zip(shape, dens)])
    grid = []
    ndim = len(shape)
    for i in range(ndim):
        sh = np.ones((ndim)).astype(int)
        sh[i] = size[i]
        x = np.arange(size[i]).reshape(sh) / dens[i]
        for j in range(ndim):
            if j != i:
                x = np.repeat(x, size[j], axis=j)
        grid.append(x)
    grid = np.stack(grid)
    return grid


def calc_worley(*shape, dens=4, padded=False, seed=None):
    """
    Calculate Worley noise with continuous boundary conditions and indices output.
    
    Parameters
    ----------
    *shape : Union[int, Tuple[int, ...]]
        Shape of the noise grid.
    dens : Union[int, Tuple[int, ...]], default 4
        Number of interpolation points between grid nodes along each axis
    padded : bool, default False
        If True - return padded centers and center indices.
    seed : int, default None
        Numpy random seed

    Returns
    -------
    dist : ndarray
        Distances array. First index corresponds to the smallest distance, second smallest, etc.
    dist_idxs : ndarray
        Center indices corresponding to each distance value.
    centers : ndarray
        Centers coordinates.
    center_idxs : ndarray
        Centers indices (continuous numbering).

    Example
    -------
    >>> import pythonperlin as pp
    >>> shape = (32,32)
    >>> dist, dist_idxs, centers, center_idxs = pp.calc_worley(shape, dens=8)
    >>> plt.imshow(dist[0].T)
    >>> plt.scatter(*centers)

    """    
    shape, dens = _shape_and_dens_to_tuples(*shape, dens=dens)
    ndim = len(shape)
    # Make padded centers
    centers, idx = _make_centers(shape, boundary=True, seed=seed)
    center_idxs = np.copy(idx)
    # Make noise
    noise = np.copy(centers)
    for i in range(ndim):
        noise = np.repeat(noise, dens[i], axis=i+1)
    # Repeat center indices
    for i in range(ndim):
        idx = np.repeat(idx, dens[i], axis=i)
    # Scale centers
    centers = np.stack([centers[i] * dens[i] for i in range(ndim)])
    # Slice centers
    if not padded:
        # Slice coordinates
        slc = [slice(None)] + [slice(1,-1)] * ndim
        centers = centers[tuple(slc)]
        # Slice indices
        slc = [slice(1,-1)] * ndim
        center_idxs = center_idxs[tuple(slc)]
    # Make grid
    grid = _make_grid(shape, dens=dens)
    # Calculate distances
    idxs = []
    dist = []
    ranges = [range(-1,2)] * ndim
    for direction in itertools.product(*ranges):
        sh = noise.shape
        slc = [slice(dens[i]*(d+2),sh[i+1]+dens[i]*(d-2)) for i, d in enumerate(direction)]
        slc = tuple([slice(None)] + slc)
        idxs.append(idx[slc[1:]])
        r = grid - noise[slc]
        r2 = np.sum(r**2, axis=0)
        dist.append(np.sqrt(r2))
    dist = np.stack(dist)
    # Sort distances
    sort_idx = np.argsort(dist, axis=0)
    dist = np.take_along_axis(dist, sort_idx, axis=0)
    # Sort indices of the center corresponding to each distance value
    dist_idxs = np.stack(idxs)
    dist_idxs = np.take_along_axis(dist_idxs, sort_idx, axis=0)
    return dist, dist_idxs, centers, center_idxs


def worley(*shape, dens=4, seed=None):
    """
    Generate Worley noise with continuous boundary conditions.
    
    Parameters
    ----------
    *shape : Union[int, Tuple[int, ...]]
        Shape of the noise grid.
    dens : Union[int, Tuple[int, ...]], default 4
        Number of interpolation points between grid nodes along each axis
    seed : int, default None
        Numpy random seed

    Returns
    -------
    dist : ndarray
        Distances array. First index corresponds to the smallest distance, second smallest, etc.
    centers : ndarray
        Centers coordinates.

    Example
    -------
    >>> import pythonperlin as pp
    >>> shape = (32,32)
    >>> dist,cenetrs = pp.worley(shape, dens=8)
    >>> plt.imshow(dist[0].T)
    >>> plt.scatter(*centers)

    """
    dist, _, centers, _ = calc_worley(*shape, dens=dens, seed=seed)
    return dist, centers





