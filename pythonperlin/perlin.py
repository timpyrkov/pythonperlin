#!/usr/bin/env python
# -*- coding: utf8 -*-

import itertools
import numpy as np
from typing import Union, Tuple, List, Optional, Callable, Any
from scipy.interpolate import interp1d

def smoothstep(x: np.ndarray) -> np.ndarray:
    """
    Apply smoothstep interpolation to input values.
    
    The smoothstep function provides a smooth transition between 0 and 1 with zero first derivatives at the boundaries.
    
    Parameters
    ----------
    x : np.ndarray
        Input values to be interpolated, typically in range [0, 1]
        
    Returns
    -------
    np.ndarray
        Interpolated values using the smoothstep function: 3x² - 2x³
    """
    return 3 * np.power(x, 2) - 2 * np.power(x, 3)


def smootherstep(x: np.ndarray) -> np.ndarray:
    """
    Apply smootherstep interpolation to input values.
    
    The smootherstep function provides an even smoother transition than smoothstep, with zero first and second derivatives at the boundaries.
    
    Parameters
    ----------
    x : np.ndarray
        Input values to be interpolated, typically in range [0, 1]
        
    Returns
    -------
    np.ndarray
        Interpolated values using the smootherstep function: 6x⁵ - 15x⁴ + 10x³
    """
    return 6 * np.power(x, 5) - 15 * np.power(x, 4) + 10 * np.power(x, 3)


def _shape_and_dens_to_tuples(
        *shape: Union[int, Tuple[int, ...]], 
        dens: Union[int, Tuple[int, ...]] = 1, 
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """
    Get shape and dens as tuples. Check if the shape and dens are valid.

    Parameters
    ----------
    *shape : Union[int, Tuple[int, ...]]
        Shape of the noise field. Can be provided as separate integers or a tuple.
    dens : Union[int, Tuple[int, ...]], default 1
        Number of points between each two gradients along an axis   

    Returns
    -------
    shape : Tuple[int, ...]
        Shape as tuple
    dens : Tuple[int, ...]
        Dens as tuple
    """
    shape = tuple(shape[0]) if isinstance(shape[0], (tuple, list)) else shape
    if not isinstance(shape, tuple):
        raise ValueError('shape must be a tuple')
    if any(s < 1 for s in shape):
        raise ValueError('shape must be 1 or greater')
    if isinstance(dens, int):
        if dens < 1:
            raise ValueError('dens must be 1 or greater')
        dens = tuple([dens] * len(shape))
    if not isinstance(dens, tuple):
        raise ValueError('dens must be int or tuple')
    if len(dens) != len(shape):
        raise ValueError('Shape and dens must have the same number of elements')
    if any(d < 1 for d in dens):
        raise ValueError('dens must be 1 or greater')
    return shape, dens


def _downsample_density(
        *shape: Union[int, Tuple[int, ...]], 
        dens: Union[int, Tuple[int, ...]], 
        nmax: int = 1000000
    ) -> Tuple[int, ...]:
    """
    Downsample density values to keep total grid points under nmax.
    Maintains density >= 2 for dimensions with initial density >= 2.

    Parameters
    ----------
    shape: Tuple[int, ...]
        Shape of the grid
    dens: Union[int, Tuple[int, ...]]
        Initial density values
    nmax: int, default 1000000
        Maximum allowed number of grid points
        
    Returns
    -------
    dens : Tuple[int, ...]
        Optimized density tuple that:
    """
    # Convert single density to tuple
    if isinstance(dens, int):
        dens = (dens,) * len(shape)
        
    # Calculate current total points
    total_points = np.prod([s * d for s, d in zip(shape, dens)])
    
    # If already under limit, return original densities
    if total_points <= nmax:
        return dens
        
    # Calculate target density that would give exactly nmax points
    # target = (nmax / prod(shape))^(1/ndim)
    target = (nmax / np.prod(shape)) ** (1/len(shape))
    
    # Find optimal divisors for each dimension
    optimized_dens = []
    for s, d in zip(shape, dens):
        if d >= 2:
            # For dimensions with initial density >= 2, find closest divisor
            # that maintains density >= 2 and minimizes deviation from target
            best_divisor = 1
            min_deviation = float('inf')
            
            # Try all possible divisors
            for divisor in range(1, d + 1):
                if d % divisor == 0:  # Must be a divisor
                    new_density = d // divisor
                    if new_density >= 2:  # Must maintain minimum density
                        deviation = abs(new_density - target)
                        if deviation < min_deviation:
                            min_deviation = deviation
                            best_divisor = divisor
                            
            optimized_dens.append(d // best_divisor)
        else:
            # For dimensions with initial density 1, just use 1
            optimized_dens.append(1)
            
    return tuple(optimized_dens)


def _upsample_density(
        arr: np.ndarray,
        shape: Tuple[int, ...],
        kind: str = 'linear'
    ) -> np.ndarray:
    """Interpolate an n-dimensional array to a new shape.
    
    Parameters
    ----------
    arr : np.ndarray
        Input array to interpolate
    shape : Tuple[int, ...]
        Desired shape of the output array
    kind : str, default 'linear'
        Type of interpolation to use (see scipy.interpolate.interp1d)
        
    Returns
    -------
    np.ndarray
        Interpolated array with shape target_shape
        
    Raises
    ------
    ValueError
        If target_shape has different number of dimensions than arr
    """
    if len(shape) != arr.ndim:
        raise ValueError(f"Target shape must have {arr.ndim} dimensions")
        
    # Create coordinate arrays for original and target shapes
    original_coords = [np.linspace(0, 1, s) for s in arr.shape]
    target_coords = [np.linspace(0, 1, s) for s in shape]
    
    # Interpolate along each axis
    result = arr.copy()
    for axis in range(arr.ndim):
        if arr.shape[axis] != shape[axis]:
            # Create interpolation function for current axis
            f = interp1d(original_coords[axis], result, kind=kind, axis=axis)
            # Interpolate to new coordinates
            result = f(target_coords[axis])
            
    return result


def _make_grads(*shape: Union[int, Tuple[int, ...]], seed: Optional[int] = None) -> np.ndarray:
    """
    Generate n-dimensional random gradient vectors of unit length.
    
    For dimensions greater than 1, the gradient vectors are normalized to unit length.
    
    Parameters
    ----------
    *shape : Union[int, Tuple[int, ...]]
        Shape of the gradient field. Can be provided as separate integers or a tuple.
    seed : Optional[int], default None
        Random seed for reproducibility
        
    Returns
    -------
    np.ndarray
        2D array of shape (N_vectors, M_coordinates) containing the gradient vectors
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


def _make_grid(*shape: Union[int, Tuple[int, ...]], dens: Union[int, Tuple[int, ...]] = 1, offset: bool = True) -> np.ndarray:
    """
    Generate a grid of minor nodes for placing interpolated gradient values.
    
    Creates a grid of points between adjacent major nodes, with optional offset to avoid overlapping with major nodes.
    
    Parameters
    ----------
    *shape : Union[int, Tuple[int, ...]]
        Shape of the major grid. Can be provided as separate integers or a tuple.
    dens : int, default 1
        Number of minor nodes between two major nodes along each axis
    offset : bool, default True
        If True, offset minor nodes by half-step to avoid overlapping with major nodes
        
    Returns
    -------
    np.ndarray
        Array of grid coordinates with shape (N_points, N_dimensions)
    """
    shape, dens = _shape_and_dens_to_tuples(*shape, dens=dens)
    grid = [np.linspace(0, s, s * d + 1)[:-1] for s, d in zip(shape, dens)]
    if offset:
        grid = [g + 0.5 * g[1] for g in grid]
    grid = list(itertools.product(*grid))
    grid = np.array(grid)
    return grid


def _get_shape(grid: np.ndarray, major_ticks: bool = False) -> Tuple[int, ...]:
    """
    Infer the shape of the grid from the grid coordinates.
    
    Parameters
    ----------
    grid : np.ndarray
        Array of grid coordinates
    major_ticks : bool, default False
        If True, infer shape of major grid nodes instead of minor nodes
        
    Returns
    -------
    Tuple[int, ...]
        Shape of the grid as a tuple of integers
    """
    shape = tuple(len(np.unique(g)) for g in grid.T)
    if major_ticks:
        shape = tuple(np.max(grid + 1, axis=0).astype(int))
    return shape


def _grid_to_grads(grid: np.ndarray, grads: np.ndarray, axes: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Get gradient vectors of the closest major node along specified axes.
    
    Parameters
    ----------
    grid : np.ndarray
        Array of minor grid coordinates
    grads : np.ndarray
        Array of gradient vectors
    axes : Optional[np.ndarray], default None
        Direction vector given by tuple of zeros and ones
        
    Returns
    -------
    np.ndarray
        Array of gradient vectors corresponding to the closest major nodes
    """
    if axes is None:
        axes = np.zeros((grid.shape[-1]))
    idx = np.floor(grid).astype(int) + axes
    shape = _get_shape(grid, major_ticks=True)
    ndim = len(shape)
    for i in range(ndim):
        idx[:,i] = (idx[:,i] % shape[i]) * np.prod(shape[i+1:])
    idx = np.sum(idx, axis=1)
    v = grads[idx]
    return v


def _grid_to_dist(grid: np.ndarray, axes: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Calculate distance vectors from the closest major node along specified axes.
    
    Parameters
    ----------
    grid : np.ndarray
        Array of minor grid coordinates
    axes : Optional[np.ndarray], default None
        Direction vector given by tuple of zeros and ones
        
    Returns
    -------
    np.ndarray
        Array of distance vectors from the closest major nodes
    """
    if axes is None:
        axes = np.zeros((grid.shape[-1]))
    r = grid - np.floor(grid)
    r = r - axes
    r = r.astype(np.float16)
    return r


def _calc_grid(grid: np.ndarray, grads: np.ndarray, smooth: Optional[Callable[[np.ndarray], np.ndarray]] = smoothstep) -> np.ndarray:
    """
    Calculate interpolated noise values for grid nodes.
    
    Parameters
    ----------
    grid : np.ndarray
        Array of minor grid coordinates
    grads : np.ndarray
        Array of gradient vectors
    smooth : Optional[Callable[[np.ndarray], np.ndarray]], default smoothstep
        Smoothing function to use for interpolation
        
    Returns
    -------
    np.ndarray
        Flattened array of interpolated noise values
    """
    noise = []
    ndim = grid.shape[-1]
    ranges = [range(2)] * ndim
    for axes in itertools.product(*ranges):
        v = _grid_to_grads(grid, grads, axes)
        r = _grid_to_dist(grid, axes)
        d = np.sum(v * r, axis=1)
        noise.append(d)
    r = _grid_to_dist(grid).T[::-1]
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


def _domain_warping(*shape: Union[int, Tuple[int, ...]], grid: Optional[np.ndarray] = None, seed: int = 0, 
                  warp: float = 0.0, smooth: Optional[Callable[[np.ndarray], np.ndarray]] = smoothstep) -> np.ndarray:
    """
    Apply domain warping to the grid coordinates.
    
    Domain warping creates more complex noise patterns by perturbing the input coordinates.
    
    Parameters
    ----------
    *shape : Union[int, Tuple[int, ...]]
        Shape of the noise field
    grid : Optional[np.ndarray], default None
        Grid coordinates to warp
    seed : int, default 0
        Random seed for gradient generation
    warp : float, default 0.0
        Magnitude of domain warping
    smooth : Optional[Callable[[np.ndarray], np.ndarray]], default smoothstep
        Smoothing function to use for interpolation
        
    Returns
    -------
    np.ndarray
        Warped grid coordinates
    """
    nnode, ndim = grid.shape
    vmax = np.max(grid, axis=0)
    for i in range(ndim):
        grads = _make_grads(*shape, seed=seed)
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


def perlin(*shape: Union[int, Tuple[int, ...]], dens: Union[int, Tuple[int, ...]] = 1, octaves: int = 0, seed: Optional[int] = None,
          warp: float = 0.0, smooth: Optional[Callable[[np.ndarray], np.ndarray]] = smoothstep) -> np.ndarray:
    """
    Generate Perlin noise with continuous boundary conditions.
    
    Parameters
    ----------
    *shape : Union[int, Tuple[int, ...]]
        Shape of the noise field. Can be provided as separate integers or a tuple.
    dens : Union[int, Tuple[int, ...]], default 1
        Number of points between each two gradients along an axis
    octaves : int, default 0
        Number of additional octaves to add for more detail
    seed : Optional[int], default None
        Random seed for gradient generation
    warp : float, default 0.0
        Magnitude of domain warping
    smooth : Optional[Callable[[np.ndarray], np.ndarray]], default smoothstep
        Smoothing function to use for interpolation
        
    Returns
    -------
    np.ndarray
        Noise grid with the specified shape
        
    Examples
    --------
    >>> from pythonperlin import perlin
    >>> shape = (32, 32)
    >>> x = perlin(shape, dens=8)
    """
    # Get shape and dens tuples
    shape, dens = _shape_and_dens_to_tuples(*shape, dens=dens)

    # Check output shape
    nmax = 150_000_000
    output_shape = tuple(d * s for d, s in zip(dens, shape))
    if np.prod(output_shape) > nmax:
        raise ValueError(f'Total number of grid points ({np.prod(output_shape)}) exceeds maximum allowed number of grid points ({nmax}).')

    # Downsample density if number of grid points is too high
    nmax, nopt = 10_000_000, 1_000_000
    dens = _downsample_density(*shape, dens=dens, nmax=nopt)
    noise_shape = tuple(d * s for d, s in zip(dens, shape))
    if np.prod(noise_shape) > nmax:
        raise ValueError(f'Cannot optimize number of grid points ({np.prod(noise_shape)}) ' \
                         f'below maximum allowed number of grid points ({nmax}). ' \
                         f'Try to reduce shape or use factorizable numbers for density, e.g. powers of 2.')

    # Calculate Perlin noise
    grid = _make_grid(*shape, dens=dens)
    if warp:
        grid = _domain_warping(*shape, grid=grid, seed=seed, warp=warp, smooth=smooth)
    grads = _make_grads(*shape, seed=seed)
    noise = _calc_grid(grid, grads, smooth=smooth)
    for i in range(octaves):
        grid = 2 * grid
        scale = 1 / 2**(i+1)
        shape = tuple(2 * s for s in shape)
        grads = _make_grads(*shape, seed=seed)
        noise += scale * _calc_grid(grid, grads, smooth=smooth)
    noise = noise.reshape(noise_shape)

    # Upsample density back to original shape if downsampling was applied
    if np.prod(noise_shape) < np.prod(output_shape):
        noise = _upsample_density(noise, output_shape, kind='linear')

    return noise






