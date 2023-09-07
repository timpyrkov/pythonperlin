#!/usr/bin/env python
# -*- coding: utf8 -*-

import numpy as np
from scipy.interpolate import interp1d
import itertools


def smoothstep(x):
    """
    Smoothstep for interpolation

    """
    return 3 * np.power(x, 2) - 2 * np.power(x, 3)


def smootherstep(x):
    """
    Smootherstep for interpolation

    """
    return 6 * np.power(x, 5) - 15 * np.power(x, 4) + 10 * np.power(x, 3)


def make_grads(*shape, seed=None):
    """
    Make n-dimensional random gradient vectors
    of unit-length for ndim > 1
    
    Parameters
    ----------
    *shape
        Integers or tuple of integers

    Returns
    -------
    grads : ndarray
        2D array N vectors x M coordinates

    """
    shape = tuple(shape[0]) if isinstance(shape[0], (tuple, list)) else shape
    ndim = len(shape)
    np.random.seed(seed)
    grads = np.random.normal(0, 1, np.prod(shape) * ndim).astype(np.float16)
    if ndim == 1:
        grads = grads.reshape(-1,1)
    else:
        grads = grads.reshape(ndim, -1)
        grads = grads / np.linalg.norm(grads, axis=0)
        grads = grads.T
    return grads


def make_grid(*shape, dens=1, offset=True):
    """
    Make minor grid nodes -- to place interpolated values
    
    Parameters
    ----------
    *shape
        Integers or tuple of integers
    dens : int, default 1
        Number of minor nodes between two major integer-positioned nodes
    offset : bool, default True
        If true, offset minor nodes by half-step to not overlap major nodes

    Returns
    -------
    grid : ndarray
        Grid float coordinates

    """
    shape = tuple(shape[0]) if isinstance(shape[0], (tuple, list)) else shape
    grid = [np.linspace(0, s, s * dens + 1)[:-1] for s in shape]
    if offset:
        grid = [g + 0.5 * g[1] for g in grid]
    grid = list(itertools.product(*grid))
    grid = np.array(grid)
    return grid


def get_shape(grid, major_ticks=False):
    """
    Infer shape from grid
    
    Parameters
    ----------
    grid : ndarray
        Minor grid nodes array
    major_ticks : bool, default False
        If true, infer shape of majr grid nodes

    Returns
    -------
    shape : tuple
        Shape of grid ndarray

    """
    shape = tuple(len(np.unique(g)) for g in grid.T)
    if major_ticks:
        shape = tuple(np.max(grid + 1, axis=0).astype(int))
    return shape


def grid_to_grads(grid, grads, axes=None):
    """
    Get gradient of the closest major node along axes
    
    Parameters
    ----------
    grid : ndarray
        Minor grid nodes array
    grads : ndarray
        2D array N vectors x M coordinates
    axes : tuple, default None
        Direction given by tuple of zeros ans ones

    Returns
    -------
    v : ndarray
        Granient vector array N vectors x M components

    """
    if axes is None:
        axes = np.zeros((grid.shape[-1]))
    idx = np.floor(grid).astype(int) + axes
    shape = get_shape(grid, major_ticks=True)
    ndim = len(shape)
    for i in range(ndim):
        idx[:,i] = (idx[:,i] % shape[i]) * np.prod(shape[i+1:])
    idx = np.sum(idx, axis=1)
    v = grads[idx]
    return v


def grid_to_dist(grid, axes=None):
    """
    Get distance from the closest major node along axes
    
    Parameters
    ----------
    grid : ndarray
        Minor grid nodes array
    grads : ndarray
        2D array N vectors x M coordinates
    axes : tuple, default None
        Direction given by tuple of zeros ans ones

    Returns
    -------
    r : ndarray
        Distance vector array N vectors x M components

    """
    if axes is None:
        axes = np.zeros((grid.shape[-1]))
    r = grid - np.floor(grid)
    r = r - axes
    r = r.astype(np.float16)
    return r


def calc_grid(grid, grads, smooth=smoothstep):
    """
    Calculate interpolated noise values for grid nodes
    
    Parameters
    ----------
    grid : ndarray
        Minor grid nodes array
    grads : ndarray
        2D array N vectors x M coordinates
    smooth : function, default smoothstep
        Smooth function or None

    Returns
    -------
    noise : ndarray
        Noise grid flattened

    """
    noise = []
    ndim = grid.shape[-1]
    ranges = [range(2)] * ndim
    for axes in itertools.product(*ranges):
        v = grid_to_grads(grid, grads, axes)
        r = grid_to_dist(grid, axes)
        d = np.sum(v * r, axis=1)
        noise.append(d)
    r = grid_to_dist(grid).T[::-1]
    for i in range(ndim):
        t = r[i]
        new_noise = []
        ranges = [range(2)] * (ndim - i)
        for axes in itertools.product(*ranges):
            fade = t if axes[-1] else 1 - t
            fade = fade if smooth is None else smooth(fade)
            d = noise.pop(0)
            new_noise.append(d * fade)
        for i in range(len(new_noise))[::2]:
            noise.append(new_noise[i] + new_noise[i + 1])
    noise = noise[0].astype(float)
    return noise


def domain_warping(*shape, grid=None, seed=0, warp=0.0, smooth=smoothstep):
    """
    Apply domain warping to grid

    Parameters
    ----------
    *shape
        Integers or tuple of integers
    grid : ndarray
        Grid float coordinates
    seed : int, default None
        Numpy random seed
    warp : float, default 0.0
        Magnitude of domain warping
    smooth : {smoothstep, smootherstep, None}, default smoothstep
        Smooth function or None

    Returns
    -------
    grid : ndarray
        Grid float coordinates

    """
    nnode, ndim = grid.shape
    vmax = np.max(grid, axis=0)
    for i in range(ndim):
        grads = make_grads(*shape, seed=seed)
        w = calc_grid(grid, grads, smooth=smooth)
        x = grid[:,i]
        vmax = int(x.max()) + 1
        x = x + warp * w
        mask = x < 0
        x[mask] = x[mask] + vmax
        mask = x >= vmax
        x[mask] = x[mask] - vmax
        grid[:,i] = x
    return grid


def perlin(*shape, dens=1, octaves=0, seed=None, warp=0.0, smooth=smoothstep):
    """
    Generate Perlin noise
    
    Parameters
    ----------
    *shape
        Integers or tuple of integers
    dens : int, default 1 (white noise)
        Number of points between each two gradients along an axis
    octaves : int, default 0
        Number of additional octaves
    seed : int, default None
        Numpy random seed
    warp : float, default 0.0
        Magnitude of domain warping
    smooth : {smoothstep, smootherstep, None}, default smoothstep
        Smooth function or None

    Returns
    -------
    noise : ndarray
        Noise grid

    Example
    -------
    >>> from pythonperlin import perlin
    >>> shape = (32,32)
    >>> x = perlin(shape, dens=8)

    """
    shape = tuple(shape[0]) if isinstance(shape[0], (tuple, list)) else shape
    noise_shape = tuple(dens * s for s in shape)
    grid = make_grid(*shape, dens=dens)
    if warp:
        grid = domain_warping(*shape, grid=grid, seed=seed, warp=warp, smooth=smooth)
    grads = make_grads(*shape, seed=seed)
    noise = calc_grid(grid, grads, smooth=smooth)
    for i in range(octaves):
        grid = 2 * grid
        scale = 1 / 2**(i+1)
        shape = tuple(2 * s for s in shape)
        grads = make_grads(*shape, seed=seed)
        noise += scale * calc_grid(grid, grads, smooth=smooth)
    noise = noise.reshape(noise_shape)
    return noise


def extend2d(x, n=1, axis=None, kind='linear', mode='full'):
    """
    Extend by inserting N new dots between grid nodes along an axis
    
    Parameters
    ----------
    x : ndarray
        Array of values
    n : int, default 1
        Number of dots to insert between grid nodes along specified axis
    axis : int or None, default None
        Specifies the axis to extend (if not specified - extend by both axes)
    kind : str, default 'linear'
        See description of 'kind' parameter for scipy.interpolate.interp1d()
    mode : {'full', 'same'}, default 'full'
        'full;'' keep all values, 'same' truncates to the original size

    Returns
    -------
    ndarray
        Array of extended values

    """
    assert x.ndim == 2
    if axis is None:
        x_ = extend2d(x, n, 0, kind, mode)
        x_ = extend2d(x_, n, 1, kind, mode)
    else:
        m = x.shape[axis]
        l = np.linspace(0, 1, m)
        f = interp1d(l, x, kind, axis)
        l = np.linspace(0, 1, (n + 1) * (m - 1) + 1)
        x_ = f(l)
        if mode == 'same':
            slc = [slice(None)] * x.ndim
            slc[axis] = slice(0, m)
            x_ = x_[tuple(slc)]
    return x_






