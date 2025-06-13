#!/usr/bin/env python
# -*- coding: utf8 -*-

import numpy as np
import itertools
import warnings
import colorsys
import matplotlib
import matplotlib.colors as mc
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, to_hex
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Union, Tuple, List, Optional, Literal
from math import sqrt, acos, pi
from scipy.spatial import Voronoi
from .perlin import _shape_and_dens_to_tuples, perlin

try:
    import plotly.graph_objects as go
    from plotly.offline import init_notebook_mode
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def adjust_color(color: str, factor: float) -> str:
    """
    Adjust color brightness using HLS color space.
    
    Parameters
    ----------
    color : str
        Color name or hex code
    factor : float
        Brightness adjustment factor (> 0 to lighten, < 0 to darken)
        
    Returns
    -------
    str
        Adjusted color in hex format
    """
    
    # Convert color to RGB
    rgb = to_rgb(color)
    
    # Convert RGB to HLS
    h, l, s = colorsys.rgb_to_hls(*rgb)

    # Adjust lightness
    l = np.clip(l + factor, 0, 1)
    
    # Convert back to RGB and then to hex
    rgb_adjusted = colorsys.hls_to_rgb(h, l, s)
    return to_hex(rgb_adjusted)


def cartesian_to_spherical(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert Cartesian coordinates to spherical coordinates.
    
    Parameters
    ----------
    x : np.ndarray
        X coordinates
    y : np.ndarray
        Y coordinates
    z : np.ndarray
        Z coordinates
        
    Returns
    -------
    r : np.ndarray
        Radius
    theta : np.ndarray
        Polar angle from z-axis
    phi : np.ndarray
        Azimuthal angle from x-axis
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z/r)  # Polar angle from z-axis
    phi = np.arctan2(y, x)  # Azimuthal angle from x-axis
    return r, theta, phi

def spherical_to_cartesian(r: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert spherical coordinates to Cartesian coordinates.
    
    Parameters
    ----------
    r : np.ndarray
        Radius
    theta : np.ndarray
        Polar angle from z-axis
    phi : np.ndarray
        Azimuthal angle from x-axis
        
    Returns
    -------
    x : np.ndarray
        X coordinates
    y : np.ndarray
        Y coordinates
    z : np.ndarray
        Z coordinates
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def _is_prime_number(n: int) -> bool:
    """
    Check if a number is prime.
    """
    if n <= 1:
        return False
    for i in range(2, int(np.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True


def _unique_tuples(list_of_tuples: List[Tuple[int, int]]) -> Tuple[List[Tuple[int, int]], List[int]]:
    """
    Return a list of unique tuples and a list of counts for each tuple.
    """
    values = list(set(list_of_tuples))
    counts = [list_of_tuples.count(v) for v in values]
    return values, counts


def _tilt_axis(x, tilt: Literal["forward", "backward", "inward", "outward"] = 'forward'):
    """
    Tilt an axis of a grid.
    """
    if x.size == 0:
        return x
    if tilt in ["forward", "inward"]:
        x[:, ::2] += 0.5
    if tilt in ["backward", "outward"]:
        x[:, 1::2] += 0.5
    if tilt == "inward":
        x[-1, ::2] = np.nan
    if tilt == "outward":
        x[-1, 1::2] = np.nan
    return x


def _reverse_tilt(tilt: Literal["forward", "backward", "inward", "outward"]):
    """
    Reverse the tilt of a grid.
    """
    if tilt == "forward":
        tilt = "backward"
    elif tilt == "backward":
        tilt = "forward"
    elif tilt == "inward":
        tilt = "outward"
    elif tilt == "outward":
        tilt = "inward"
    return tilt


def _outside_box(box, x, y, epsilon=0.1):
    """
    Check if a point is outside a box.
    """
    x_min, x_max = box[0][0] + epsilon, box[1][0] - epsilon
    y_min, y_max = box[0][1] + epsilon, box[1][1] - epsilon
    if x < x_min or x > x_max or y < y_min or y > y_max:
        return True
    else:
        return False


def _common_divisors(a: int, b: int) -> List[int]:
    """
    Find common divisors of two integers.
    """
    divisors = [i for i in range(1, min(a, b)) if a % i == 0 and b % i == 0]
    return divisors


def _hex_to_triang_size(size, base, tilt):
    """
    Convert hexagonal grid size to triangular grid size.
    """
    nx, ny = size[:2]
    base_triang = base
    tilt_triang = tilt
    if base == "horizontal":
        nx_triang = 3 * (nx // 2) + 2 * int(nx % 2 == 1)
        ny_triang = 2 * ny + int(tilt in ["forward", "backward"])
        if tilt == "forward":
            if nx % 2 != 0:
                tilt_triang = "inward"
        elif tilt == "backward":
            if nx % 2 != 0:
                tilt_triang = "outward"
        elif tilt == "inward":
            if nx % 2 == 0:
                tilt_triang = "backward"
            else:
                tilt_triang = "outward"
        elif tilt == "outward":
            if nx % 2 == 0:
                tilt_triang = "forward"
            else:
                tilt_triang = "inward"
    elif base == "vertical":
        nx_triang = 2 * nx + int(tilt in ["forward", "backward"])
        ny_triang = 3 * (ny // 2) + 2 * int(ny % 2 == 1)
        if tilt == "forward":
            if ny % 2 != 0:
                tilt_triang = "inward"
        elif tilt == "backward":
            if ny % 2 != 0:
                tilt_triang = "outward"
        elif tilt == "inward":
            if ny % 2 == 0:
                tilt_triang = "backward"
            else:
                tilt_triang = "outward"
        elif tilt == "outward":
            if ny % 2 == 0:
                tilt_triang = "forward"
            else:
                tilt_triang = "inward"
    size_triang = (nx_triang, ny_triang)
    return size_triang, base_triang, tilt_triang


def _hex_to_noise_shape_and_dens(size_triang, shape, dens, tile):
    """
    Convert hexagonal grid shape / density to triangular grid shape / density.
    """
    # Calculate dens_triang for Perlin noise
    shape_triang = list(shape)
    dens_triang = list(dens)
    for axis in [0, 1]:
        # Find divisors of size_triang[axis]
        divisors = []
        for i in range(1, size_triang[axis] + 1):
            if size_triang[axis] % i == 0:
                divisors.append(i)
        # Find the closest divisor that is approximately 2-3 times dens[axis]
        target_dens = dens[axis] * 2.5  # Target is 2.5 times original density
        closest_divisor = min(divisors, key=lambda x: abs(x - target_dens))
        if tile:
            dens_triang[axis] = closest_divisor
            shape_triang[axis] = size_triang[axis] // closest_divisor
        else:
            dens_triang[axis] = int(target_dens)
            shape_triang[axis] = int(np.ceil(size_triang[axis] / dens_triang[axis]))
    shape_triang = tuple(shape_triang)
    dens_triang = tuple(dens_triang)
    return shape_triang, dens_triang


def _hex_to_triang_nan(x, tilt, base):
    """
    Convert hexagonal grid to triangular grid and set NaNs.
    """
    if tilt == "forward":
        i0, i1 = 2, 1
    elif tilt == "backward":
        i0, i1 = 1, 2
    elif tilt == "inward":
        i0, i1 = 1, 1
    elif tilt == "outward":
        i0, i1 = 2, 2
    if base == "horizontal":
        n = x.shape[0]
        idxs = np.arange(n)[i0::3]
        x[idxs, 0] = np.nan
        if i0 == 1:
            x[0,0] = np.nan
        if idxs[-1] == n-2:
            x[n-1,0] = np.nan
        idxs = np.arange(n)[i1::3]
        x[idxs, -1] = np.nan
        if i1 == 1:
            x[0, -1] = np.nan
        if idxs[-1] == n-2:
            x[n-1, -1] = np.nan
    elif base == "vertical":
        n = x.shape[1]
        idxs = np.arange(n)[i0::3]
        x[0, idxs] = np.nan
        if i0 == 1:
            x[0,0] = np.nan
        if idxs[-1] == n-2:
            x[0, n-1] = np.nan
        idxs = np.arange(n)[i1::3]
        x[-1, idxs] = np.nan
        if i1 == 1:
            x[-1,0] = np.nan
        if idxs[-1] == n-2:
            x[-1, n-1] = np.nan
    return x


def _create_padding_mask(x, triangles, is_padding):
    """
    Create a mask for vertices that belong to padding triangles.
    """
    mask = np.zeros(x.shape, dtype=bool)
    for coords, ispad in zip(triangles, is_padding):
        if not ispad:
            for (i, j) in coords:
                mask[i, j] = True

    return mask


def _tile_noise(noise: np.ndarray, i1: int, i2: int, axis: int) -> np.ndarray:
    """
    Helper function to pad noise array along specified axis.
    """
    # Initialize output
    output = noise.copy()
    
    # Add padding at the start
    if i1 > 0:
        nfull = i1 // noise.shape[axis]
        npart = i1 % noise.shape[axis]
        if axis == 0:
            part = noise[-npart:] if npart > 0 else noise[:0]
            pad = np.concatenate([part] + [noise] * nfull, axis=0)
        else:
            part = noise[:,-npart:] if npart > 0 else noise[:, :0]
            pad = np.concatenate([part] + [noise] * nfull, axis=1)
        output = np.concatenate((pad, noise), axis=axis)
    
    # Add padding at the end
    if i2 > 0:
        nfull = i2 // noise.shape[axis]
        npart = i2 % noise.shape[axis]
        if axis == 0:
            part = noise[:npart]
            pad = np.concatenate([noise] * nfull + [part], axis=0)
        else:
            part = noise[:, :npart]
            pad = np.concatenate([noise] * nfull + [part], axis=1)
        output = np.concatenate((output, pad), axis=axis)
    
    return output


def _transpose_face_idxs(face_idxs: List[List[int]]) -> List[List[int]]:
    """
    Transpose the face indices (y,x) -> (x,y).
    """
    new_idxs = []
    for face in face_idxs:
        new_idxs.append([(j, i) for i, j in face])
    return new_idxs


def _calc_normals(x: np.ndarray, y: np.ndarray, z: np.ndarray, faces: List[List[int]]) -> np.ndarray:
    """
    Calculate normal vectors for each face.
    
    Parameters
    ----------
    x : np.ndarray
        X coordinates
    y : np.ndarray
        Y coordinates
    z : np.ndarray
        Z coordinates
    faces : List[List[int]]
        List of face vertex indices
        
    Returns
    -------
    np.ndarray
        Array of face normal vectors
    """
    normals = []
    for face in faces:
        # Get vertex indices
        i = face[0]
        j = face[1]
        k = face[2]

        # Get vertices of the face
        v1 = np.array([x[i], y[i], z[i]])
        v2 = np.array([x[j], y[j], z[j]])
        v3 = np.array([x[k], y[k], z[k]])
        
        # Calculate two edge vectors
        # Assume counter-clockwise winding
        edge1 = v2 - v1
        edge2 = v3 - v2
        
        # Calculate normal using cross product
        normal = np.cross(edge1, edge2)
        
        # Normalize
        normal = normal / np.linalg.norm(normal)
        normals.append(normal)
    
    return np.array(normals)


def _adjust_lighting(
    face_normals: np.ndarray,
    light_source: np.ndarray = np.array([1,1,1]),
    view_vector: np.ndarray = np.array([0,0,1]),
    base_color: Union[str, List[str]] = "grey",
    ambient_intensity: float = 0.1,
    diffuse_intensity: float = 0.2,
    specular_intensity: float = 0.1,
    shininess: float = 10.0
) -> List[str]:
    """
    Calculate lighting effects for each face.
    
    Parameters
    ----------
    face_normals : np.ndarray
        Array of face normal vectors
    light_source : np.ndarray, default np.array([1,1,1])
        Normalized light source direction
    view_vector : np.ndarray, default np.array([0,0,1])
        Normalized view direction
    base_color : str, default "grey"
        Base color for faces
    ambient_intensity : float, default 0.1
        Intensity of ambient light
    diffuse_intensity : float, default 0.2
        Intensity of diffuse light
    specular_intensity : float, default 0.1
        Intensity of specular light
    shininess : float, default 10.0
        Surface shininess factor
        
    Returns
    -------
    List[str]
        List of adjusted colors for each face
    """
    # Normalize light source and view vector
    light_source = np.array(light_source)
    light_source = light_source / np.linalg.norm(light_source)
    view_vector = np.array(view_vector)
    view_vector = view_vector / np.linalg.norm(view_vector)
    
    face_colors = []
    for i,normal in enumerate(face_normals):
        # Calculate diffuse lighting
        diffuse = diffuse_intensity * np.dot(normal, light_source)
        
        # Calculate specular lighting
        reflection = max(0, np.dot(normal, light_source))
        reflection = 2 * reflection * normal - light_source
        specular = specular_intensity * max(0, np.dot(reflection, view_vector)) ** shininess
        
        # Combine lighting components
        intensity = ambient_intensity + diffuse + specular
        intensity = np.clip(intensity, -1, 1)
        
        # Adjust base color based on lighting
        color = base_color if isinstance(base_color, str) else base_color[i]

        color = adjust_color(color, intensity)
        face_colors.append(color)
    
    return face_colors


def _spherical_kernels(n_points, dens, scale=1.0):
    """
    Calculate latitude kernels for each layer of the icosahedron.
    
    Parameters
    ----------
    n_points : np.ndarray
        Number of points per layer
    dens : int
        Density of the icosahedron
    scale : float, default 1.0
        Scale factor for the kernel
        
    Returns
    -------
    kernel : List[np.ndarray]
        List of kernels for each layer
    """
    n_max = max(n_points)
    n_min = min(n_points[1:-1])
    kernel = []
    for i, n in enumerate(n_points):
        if n == 1 or n == n_max:
            kernel.append(np.ones((1,)))
        else:
            w_max = min(n_points[i], len(n_points))
            w_max = w_max if w_max % 2 == 1 else w_max - 1
            w = dens - scale * n * (dens - 1) / (n_max - n_min)
            w = 2 * int(max(1, w) // 2) + 1
            w = min(w, w_max)
            if w >= 3:
                w = np.bartlett(w)
                w = w / np.sum(w)
                kernel.append(w)
            else:
                kernel.append(np.ones((1,)))
    return kernel


def _latitude_smoothing(x, kernel):
    """
    Smooth the latitude of the icosahedron while preserving signal amplitude.
    
    Parameters
    ----------
    x : np.ndarray
        Input signal to smooth
    kernel : np.ndarray
        Smoothing kernel
        
    Returns
    -------
    np.ndarray
        Smoothed signal with preserved amplitude
    """
    # Create padding
    n = min(len(x) - 1, len(kernel))
    x_ = x[:-1]
    pad0 = x_[-n:]
    pad1 = x_[:n]
    x_pad = np.concatenate([pad0, x_, pad1])
    
    # Smooth
    x_smooth = np.convolve(x_pad, kernel, mode='same')[n:1-n]
    
    # Preserve signal amplitude by scaling to match original quantiles
    if len(x_) > 5:  # Only scale if we have enough points
        q_low, q_high = np.nanquantile(x_, [0.2, 0.8])
        q_low_smooth, q_high_smooth = np.nanquantile(x_smooth, [0.2, 0.8])

        # Apply linear scaling to match original quantiles
        scale = (q_high - q_low) / (q_high_smooth - q_low_smooth) if q_high_smooth != q_low_smooth else 1.0
        offset = q_low - q_low_smooth * scale
        x_smooth = x_smooth * scale + offset
    
    return x_smooth


def _longitude_idxs(n_points):
    """
    Calculate indices of the longitudes in each latitude line.
    
    Parameters
    ----------
    n_points : np.ndarray
        Number of points at each latitude
        
    Returns
    -------
    List[np.ndarray]
        List of indices of the longitudes in each latitude line
    """
    idxs = []
    n_max = max(n_points)
    idx = np.arange(n_max)
    idxs.append(idx)
    while len(idx) > 1:
        m = int(len(idx) / 5)
        mask = np.ones((len(idx))).astype(bool)
        mask[1::m] = False
        idx = idx[mask]
        idxs.append(idx)
    n_middle = len(n_points) - 2 * len(idxs)
    idxs = idxs[::-1] + [idxs[0]] * n_middle + idxs
    idxs = [np.pad(idx, (0, n_max - len(idx)), mode='constant', constant_values=-1) for idx in idxs]
    idxs = np.array(idxs)
    return idxs


def _longitude_smoothing(x, n_points, kernels):
    """
    Smooth the spherical grid along longitudes, accounting for the variable
    number of points at each latitude.
    
    Parameters
    ----------
    x : np.ndarray
        2D grid with values arranged by [latitude, longitude]
    n_points : np.ndarray
        Number of valid points at each latitude
    kernels : List[np.ndarray]
        List of smoothing kernels for each latitude
    
    Returns
    -------
    np.ndarray
        Grid with smoothed values along longitudes
    """
    # Calculate longitude indices
    idxs = _longitude_idxs(n_points)
    n_row, n_col = idxs.shape

    # Initialize output array
    x_smooth = np.zeros((n_row, n_col)) * np.nan
    
    # Process each latitude
    for k in range(n_col):

        # Collect values for this longitude
        j = np.argmin(np.abs(idxs - k), axis=1)
        value = np.array([x[i, j[i]] for i in range(n_row)])

        # Apply mirror-style padding
        padded = np.concatenate([value[::-1], value, value[::-1]])

        # Apply point-wise smoothing
        for i in range(n_row):
            if idxs[i, j[i]] != k:
                continue
            kernel = kernels[i]
            if len(kernel) < 3:
                x_smooth[i, j[i]] = value[i]
            else:
                i0 = n_row + i - len(kernel) // 2
                i1 = i0 + len(kernel)
                s = np.convolve(padded[i0:i1], kernel, mode='same')
                x_smooth[i, j[i]] = s[len(kernel) // 2 + 1]
    
    return x_smooth


def _make_icosahedron(n: int = 1, nmax: int = 6) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[List[int]]]:
    """
    Create an icosahedron with optional subdivision.
    
    Parameters
    ----------
    n : int, default 1
        Number of subdivisions
    nmax : int, default 6
        Maximum number of icosahedron subdivisions to save resources while changing shape and dens.
        
    Returns
    -------
    x : np.ndarray
        X coordinates
    y : np.ndarray
        Y coordinates
    z : np.ndarray
        Z coordinates
    faces : List[List[int]]
        List of faces (each face is a list of 3 vertex indices)

    """
    # Limit the number of subdivisions to 6 to save computational resources
    if not 1 <= n <= nmax:
        raise ValueError(f"Subdivision level n must be between 1 and {nmax}")
    
    # Calculate number of layers and max number of points per layer
    n_layers = 2 ** (n + 1) - 2 ** (n - 1) + 1
    max_points = 5 * 2 ** (n - 1) + 1

    # Calculate number of points per each layer
    n_points = np.array([np.clip(5 * i, 1, max_points - 1) for i in range(n_layers)])
    n_points = np.min(np.stack([n_points, n_points[::-1]]), axis=0) + 1
    n_points[0] = n_points[-1] = 1
    

    # Initialize spherical coordinates
    phi = np.full((n_layers, max_points), np.nan)
    theta = np.full((n_layers, max_points), np.nan)
    r = np.full((n_layers, max_points), np.nan)

    # Calculate phi_0 for each layer
    n_cap = np.where(n_points == max_points)[0].min()
    n_belt = n_layers - 2 * n_cap - 1
    phi_0 = np.array([(i - n_cap) / (n_belt) for i in range(n_layers)]) * np.pi / 5
    phi_0[:n_cap] = 0
    phi_0[-n_cap:] = np.pi / 5
    phi_0[0] = np.pi
    phi_0[-1] = np.pi

    # Fill phi array
    for i in range(n_layers):
        phi[i, :n_points[i]] = phi_0[i] + 2 * np.pi * np.arange(n_points[i]) / max(1, n_points[i] - 1)

    # Fill theta array
    for i in range(n_layers):
        theta[i, :n_points[i]] = np.pi * i / (n_layers - 1)

    # Fill r array
    for i in range(n_layers):
        r[i, :n_points[i]] = 1

    # Generate faces
    faces = []
    
    # 1. North pole cap faces (5 triangles)
    for j in range(5):
        faces.append([(0, 0), (1, j), (1, j + 1)])

    # 2. North cap layers
    for i in range(1, n_cap):        
        top_idx = [j for j in range(n_points[i+1]) if (j) % (i+1) != 0]
        base_idx = np.arange(n_points[i]).tolist()
        for j in range(len(base_idx) - 1):
            faces.append([(i, base_idx[j]), (i+1, top_idx[j]), (i, base_idx[j+1])])
        top_idx = [[j,j] if j % i == 0 else [j] for j in range(n_points[i])]
        top_idx = list(itertools.chain(*top_idx))[1:-1]
        base_idx = np.arange(n_points[i+1]).tolist()
        for j in range(len(base_idx) - 1):
            faces.append([(i+1, base_idx[j]), (i+1, base_idx[j+1]), (i, top_idx[j])])

    # 3. Middle band faces
    for i in range(n_cap, n_layers - n_cap - 1):
        base_idx = np.arange(n_points[i]).tolist()
        for j in range(len(base_idx) - 1):
            faces.append([(i,base_idx[j]), (i+1, base_idx[j]), (i, base_idx[j+1])])
            faces.append([(i,base_idx[j+1]), (i+1,base_idx[j]), (i+1,base_idx[j+1])])

    # 4. South cap layers
    for i in range(n_layers - n_cap - 1, n_layers - 2):
        top_idx = [[j,j] if j % (n_layers - i - 2) == 0 else [j] for j in range(n_points[i+1])]
        top_idx = list(itertools.chain(*top_idx))[1:-1]
        base_idx = np.arange(n_points[i]).tolist()
        for j in range(len(base_idx) - 1):
            faces.append([(i, base_idx[j]), (i+1, top_idx[j]), (i, base_idx[j+1])])
        top_idx = [j for j in range(n_points[i]) if (j) % (n_layers - i - 1) != 0]
        base_idx = np.arange(n_points[i+1]).tolist()
        for j in range(len(base_idx) - 1):
            faces.append([(i+1, base_idx[j]), (i+1, base_idx[j+1]), (i, top_idx[j])])

    # 1. North pole cap faces (5 triangles)
    for j in range(5):
        faces.append([(n_layers - 1, 0), (n_layers - 2, j + 1), (n_layers - 2, j)])

    return r, theta, phi, faces  


def suggest_shape_and_dens(
    target_size: Tuple[int, int] = (12, 10),
    grid_type: Literal["rectangular", "triangular", "hexagonal", "voronoi"] = "rectangular",
    nmax: int = 2000,
) -> None:
    """
    Suggest possible combinations of shape and density parameters 
    for "rectangular", "triangular", and "hexagonal" grids.

    Notes
    -----
    Only uses the input size as an approximate reference., and applies constaraints on
    tiling, completeness, and number of shape/dens combinations to print suggested variants.

    Parameters
    ----------
    target_size : Tuple[int, int], default (12, 10)
        Desired approximate number of grid nodes in 2D plane
    grid_type : Literal["rectangular", "triangular", "hexagonal", "voronoi"], default "rectangular"
        Type of grid to generate suggestions for
    nmax: int, default 2000
        Maximum number of grid nodes in 2D plane
    """
    if target_size[0] < 1 or target_size[1] < 1:
        raise ValueError("Target size must be positive integers.")

    # Adjust target size to the nmax
    nx, ny = target_size
    aspect_ratio = float(nx) / float(ny)
    if nx * ny > nmax:
        nx = int(np.sqrt(nmax * aspect_ratio))
        ny = int(nx / aspect_ratio)
        nx -= 3
        ny -= 3
    target_size = (nx, ny)

    # Initialize ranges for x and y
    x_range = max(4, target_size[0]) + np.arange(-3,4)
    y_range = max(4, target_size[1]) + np.arange(-3,4)

    print(f"\nSuggesting shape/density combinations for {grid_type} grid")

    # Suggest sizes for "rectangular" and "voronoi" grids
    if grid_type in ["rectangular", "voronoi"]:
        combinations = []

        # Try different shape/dens combinations
        for nx in x_range:
            for ny in y_range:
                # Skip prime numbers
                if nx > 10 and _is_prime_number(nx):
                    continue
                if ny > 10 and _is_prime_number(ny):
                    continue
                # Find dens as common divisors of nx and ny
                x_divisors = [i for i in range(1, nx) if nx % i == 0]
                y_divisors = [i for i in range(1, ny) if ny % i == 0]
                denses = list(set(x_divisors).intersection(y_divisors))
                for dens in denses:
                    shape = (nx // dens, ny // dens)
                    combinations.append({"size": (nx,ny), "shape": shape, "dens": dens})
        
        # Select size with maximum number of base/tilt combinations
        sizes = [comb["size"] for comb in combinations]
        if len(sizes) > 0:
            # Select sizes with multiple combinations
            size_list, size_count = _unique_tuples(sizes)
            if max(size_count) > 1:
                sizes = [size_list[k] for k in range(len(size_list)) if size_count[k] > 1]
            # Select size with best fill area match to the canvas
            aspect = target_size[0] / target_size[1]
            mismatch = []
            for size in sizes:
                mismatch.append(abs(aspect - size[0] / size[1]))
            size = sizes[np.argmin(mismatch)]

            # Print shape/dens combinations for the selected size
            print()
            print("-" * 40)
            print()
            for comb in combinations:
                if comb["size"] == size:
                    prtstr = f'  shape=({comb["shape"][0]}, {comb["shape"][1]})'
                    prtstr += f', dens={comb["dens"]}'
                    print(prtstr)
            print()

    # Suggest sizes for "triangular" and "hexagonal" grids
    elif grid_type in ["triangular", "hexagonal"]:
        base_tilt_combinations = [
            ("horizontal", "forward"),
            ("horizontal", "backward"),
            ("horizontal", "inward"),
            ("horizontal", "outward"),
            ("vertical", "forward"),
            ("vertical", "backward"),
            ("vertical", "inward"),
            ("vertical", "outward"),
        ]
        for base, tilt in base_tilt_combinations:
            combinations = []

            # Try different shape/dens combinations
            for nx in x_range:
                for ny in y_range:
                    # Skip prime numbers
                    if nx > 10 and _is_prime_number(nx):
                        continue
                    if ny > 10 and _is_prime_number(ny):
                        continue
                    # Skip prime numbers for hexagonal grid
                    if grid_type == "hexagonal" and tilt in ["forward", "backward"]:
                        size_triang, _, _ = _hex_to_triang_size((nx, ny), base, tilt)
                        if size_triang[0] > 10 and _is_prime_number(size_triang[0]):
                            continue
                        if size_triang[1] > 10 and _is_prime_number(size_triang[1]):
                            continue
                    # Skip non-tilable sizes
                    if grid_type == "triangular" and tilt in ["forward", "backward"]:
                        if base == "horizontal" and ny % 2 == 1:
                            continue
                        if base == "vertical" and nx % 2 == 1:
                            continue
                    if grid_type == "hexagonal" and tilt in ["forward", "backward"]:
                        if base == "horizontal" and nx % 2 == 1:
                            continue
                        if base == "vertical" and ny % 2 == 1:
                            continue
                    # Skip uncomplete sizes for hexagonal grid
                    if grid_type == "hexagonal" and tilt in ["inward", "outward"]:
                        if base == "horizontal" and nx % 2 == 0:
                            continue
                        if base == "vertical" and ny % 2 == 0:
                            continue
                    # Find dens as common divisors of nx and ny
                    x_divisors = [i for i in range(1, nx) if nx % i == 0]
                    y_divisors = [i for i in range(1, ny) if ny % i == 0]
                    denses = list(set(x_divisors).intersection(y_divisors))
                    for dens in denses:
                        shape = (nx // dens, ny // dens)
                        combinations.append({"size": (nx,ny), "shape": shape, "dens": dens, "base": base, "tilt": tilt})

            # Select size with maximum number of base/tilt combinations
            sizes = [comb["size"] for comb in combinations]
            if len(sizes) > 0:
                size_list, size_count = _unique_tuples(sizes)
                size = size_list[np.argmax(size_count)]

                # Print shape/dens combinations for the selected size
                print()
                print("-" * 40)
                print(f"{base.upper()} {tilt.upper()}")
                print()
                for comb in combinations:
                    if comb["size"] == size:
                        prtstr = f'  shape=({comb["shape"][0]}, {comb["shape"][1]})'
                        prtstr += f', dens={comb["dens"]}'
                        prtstr += f', base="{comb["base"]}", tilt="{comb["tilt"]}"'
                        print(prtstr)
                print()

    return


def suggest_shape_for_canvas(
    canvas: Tuple[int, int],
    nx: Optional[int] = None,
    ny: Optional[int] = None,
    grid_type: Literal["rectangular", "triangular", "hexagonal", "voronoi"] = "rectangular",
    nmax: int = 2000,
) -> None:
    """
    Suggest grid shape parameters based on canvas dimensions and optionally
    desired number of nodes along specified axis.

    Notes
    -----
    Only uses the input nx and ny as an approximate references., and applies constaraints on
    tiling, completeness, and number of shape/dens combinations to print suggested variants.

    Parameters
    ----------
    canvas : Tuple[int, int]
        Width and height of the canvas in pixels/units. Positive integers.
    nx : Optional[int], default None
        Desired approximate number of nodes along x axis. Positive integer.
    ny : Optional[int], default None
        Desired approximate number of nodes along y axis. Positive integer.
    grid_type : Literal["rectangular", "triangular", "hexagonal", "voronoi"], default "rectangular"
        Type of grid to generate suggestions for
    nmax: int, default 2000
        Maximum number of grid nodes in 2D plane
    """
    if canvas[0] < 1 or canvas[1] < 1:
        raise ValueError("Canvas dimensions must be positive integers.")

    if nx is not None and nx < 1:
        raise ValueError("nx must be a positive integer.")
    
    if ny is not None and ny < 1:
        raise ValueError("ny must be a positive integer.")

    # Combinations for "rectangular" and "voronoi" grids
    if grid_type in ["rectangular", "voronoi"]:

        # Flexibility to match the target size
        nflex = 3

        # Adjust target size to the canvas aspect ratio
        aspect_ratio = float(canvas[0]) / float(canvas[1])
        if nx is None and ny is None:
            denses = _common_divisors(canvas[0], canvas[1])
            denses = [d for d in denses if d > 1 and d < min(canvas[0], canvas[1])]
            denses = [d for d in denses if (canvas[0] * canvas[1]) / (d * d) <= nmax]
            if len(denses) > 0:
                dens = denses[-2:][0]
                nx = canvas[0] // dens
                ny = canvas[1] // dens
                nflex = 0
            else:
                nx = 10
                ny = max(1, int(nx / aspect_ratio))
        elif nx is None:
            nx = max(1, int(ny * aspect_ratio))
        elif ny is None:
            ny = max(1, int(nx / aspect_ratio))
        else:
            n2 = nx * ny
            nx = max(1, int(np.sqrt(n2 * aspect_ratio)))
            ny = max(1, int(nx / aspect_ratio))
        target_size = (nx, ny)

        # Adjust target size to the nmax
        nx, ny = target_size
        aspect_ratio = float(nx) / float(ny)
        if nx * ny > nmax:
            nx = int(np.sqrt(nmax * aspect_ratio))
            ny = int(nx / aspect_ratio)
            nx -= 3
            ny -= 3
        target_size = (nx, ny)

        # Initialize ranges for x and y
        x_range = max(4, target_size[0]) + np.arange(-nflex,nflex+1)
        y_range = max(4, target_size[1]) + np.arange(-nflex,nflex+1)

        # Try different shape/dens combinations
        combinations = []
        for nx in x_range:
            for ny in y_range:
                # Skip prime numbers
                if nx > 10 and _is_prime_number(nx) and len(x_range) > 1:
                    continue
                if ny > 10 and _is_prime_number(ny) and len(y_range) > 1:
                    continue
                # Calculate stride
                stride = int(min(canvas[0] / nx, canvas[1] / ny))
                # Find dens as common divisors of nx and ny
                denses = _common_divisors(nx, ny)
                for dens in denses:
                    shape = (nx // dens, ny // dens)
                    combinations.append({"size": (nx,ny), "shape": shape, "dens": dens, "stride": stride})
        
        # Select size with maximum number of base/tilt combinations and best aspect ratio match to the canvas
        sizes = [comb["size"] for comb in combinations]
        if len(sizes) > 0:
            # Select sizes with multiple combinations
            size_list, size_count = _unique_tuples(sizes)
            if max(size_count) > 1:
                sizes = [size_list[k] for k in range(len(size_list)) if size_count[k] > 1]
            # Select size with best fill area match to the canvas
            aspect = target_size[0] / target_size[1]
            mismatch = []
            for size in sizes:
                mismatch.append(abs(aspect - size[0] / size[1]))
            size = sizes[np.argmin(mismatch)]

            # Print shape/dens combinations for the selected size
            print(f'\n"{grid_type.title()}" grid options:')
            print()
            print("-" * 40)
            print()
            for comb in combinations:
                if comb["size"] == size:
                    prtstr = f'  shape=({comb["shape"][0]}, {comb["shape"][1]})'
                    prtstr += f', dens={comb["dens"]}, stride={comb["stride"]}'
                    print(prtstr)
            print()

    # Suggest sizes for "triangular" and "hexagonal" grids
    elif grid_type in ["triangular", "hexagonal"]:

        # Store input nx and ny
        input_nx = nx
        input_ny = ny

        base_tilt_combinations = [
            ("horizontal", "forward"),
            ("horizontal", "backward"),
            ("horizontal", "inward"),
            ("horizontal", "outward"),
            ("vertical", "forward"),
            ("vertical", "backward"),
            ("vertical", "inward"),
            ("vertical", "outward"),
        ]
        for base, tilt in base_tilt_combinations:

            # Adjust target size to the canvas aspect ratio
            aspect_ratio = float(canvas[0]) / float(canvas[1])
            ratio = np.sqrt(3) / 2 if grid_type == "triangular" else 2 / np.sqrt(3)
            aspect_ratio = aspect_ratio * ratio if base == "horizontal" else aspect_ratio / ratio

            if input_nx is None and input_ny is None:
                nx = 10
                ny = max(1, int(nx / aspect_ratio))
            elif input_nx is None:
                ny = input_ny
                nx = max(1, int(ny * aspect_ratio))
            elif input_ny is None:
                nx = input_nx
                ny = max(1, int(nx / aspect_ratio))
            else:
                n2 = input_nx * input_ny
                nx = max(1, int(np.sqrt(n2 * aspect_ratio)))
                ny = max(1, int(nx / aspect_ratio))
            target_size = (nx, ny)

            # Adjust target size to the nmax
            nx, ny = target_size
            aspect_ratio = float(nx) / float(ny)
            if nx * ny > nmax:
                nx = int(np.sqrt(nmax * aspect_ratio))
                ny = int(nx / aspect_ratio)
                nx -= 3
                ny -= 3
            target_size = (nx, ny)

            # Initialize ranges for x and y
            x_range = max(4, target_size[0]) + np.arange(-3,4)
            y_range = max(4, target_size[1]) + np.arange(-3,4)

            # Try different shape/dens combinations
            combinations = []
            for nx in x_range:
                for ny in y_range:
                    # Skip prime numbers
                    if nx > 10 and _is_prime_number(nx):
                        continue
                    if ny > 10 and _is_prime_number(ny):
                        continue
                    # Skip prime numbers for hexagonal grid
                    if grid_type == "hexagonal":
                        size_triang, _, _ = _hex_to_triang_size((nx, ny), base, tilt)
                        if size_triang[0] > 10 and _is_prime_number(size_triang[0]):
                            continue
                        if size_triang[1] > 10 and _is_prime_number(size_triang[1]):
                            continue
                    # Skip non-tilable sizes
                    if grid_type == "triangular" and tilt in ["forward", "backward"]:
                        if base == "horizontal" and ny % 2 == 1:
                            continue
                        if base == "vertical" and nx % 2 == 1:
                            continue
                    if grid_type == "hexagonal" and tilt in ["forward", "backward"]:
                        if base == "horizontal" and nx % 2 == 1:
                            continue
                        if base == "vertical" and ny % 2 == 1:
                            continue
                    # Skip uncomplete sizes for hexagonal grid
                    if grid_type == "hexagonal" and tilt in ["inward", "outward"]:
                        if base == "horizontal" and nx % 2 == 0:
                            continue
                        if base == "vertical" and ny % 2 == 0:
                            continue
                    # Find dens as common divisors of nx and ny
                    x_divisors = [i for i in range(1, nx) if nx % i == 0]
                    y_divisors = [i for i in range(1, ny) if ny % i == 0]
                    denses = list(set(x_divisors).intersection(y_divisors))
                    for dens in denses:
                        # Calculate shape
                        shape = (nx // dens, ny // dens)
                        # Calculate stride
                        ngrid = (nx, ny)
                        scale = 2 / np.sqrt(3)
                        if grid_type == "hexagonal":
                            ngrid, _, _ = _hex_to_triang_size(ngrid, base, tilt)
                        stride = min(int(canvas[0] / ngrid[0]), int(scale * canvas[1] / ngrid[1]))
                        if base == "vertical":
                            stride = min(int(scale * canvas[0] / ngrid[0]), int(canvas[1] / ngrid[1]))
                        combinations.append({"size": (nx,ny), "shape": shape, "dens": dens, "base": base, "tilt": tilt, "stride": stride})

            # Select size with maximum number of base/tilt combinations
            sizes = [comb["size"] for comb in combinations]
            if len(sizes) > 0:
                # Select sizes with multiple combinations
                size_list, size_count = _unique_tuples(sizes)
                if max(size_count) > 1:
                    sizes = [size_list[k] for k in range(len(size_list)) if size_count[k] > 1]
                # Select size with best fill area match to the canvas
                aspect = target_size[0] / target_size[1]
                mismatch = []
                for size in sizes:
                    mismatch.append(abs(aspect - size[0] / size[1]))
                size = sizes[np.argmin(mismatch)]

                # Print shape/dens combinations for the selected size
                print()
                print("-" * 40)
                print(f"{base.upper()} {tilt.upper()}")
                print()
                for comb in combinations:
                    if comb["size"] == size:
                        prtstr = f'  shape=({comb["shape"][0]}, {comb["shape"][1]})'
                        prtstr += f', dens={comb["dens"]}, base="{comb["base"]}"'
                        prtstr += f', tilt="{comb["tilt"]}", stride={comb["stride"]}'
                        print(prtstr)
                print()

    return


class SurfaceGrid:
    """
    A class for generating and managing surface grids with optional time dimension.
    
    This class provides functionality for creating rectangular, triangular,
    or icosahedral grids, as well as analyzing their properties.
    
    Attributes
    ----------
    _grid : Optional[np.ndarray]
        Coordinates of the grid. None if no grid has been created yet.
    _shape : Optional[Union[int, Tuple[int, ...]]]
        Shape of the grid. None if no grid has been created yet.
    _dens : Optional[int]
        Density of minor nodes. None if no grid has been created yet.
    _offset : Optional[bool]
        Whether the grid uses offset nodes. None if no grid has been created yet.
    _grid_type : Optional[Literal["rectangular", "triangulated", "icosahedron"]]
        Type of the grid. None if no grid has been created yet.
    _major : Optional[np.ndarray]
        Mask for major grid nodes. None if no grid has been created yet.
    _faces : Optional[List[Tuple[int, int, int]]]
        Faces of the grid triangulation. None if no triangulation has been calculated yet.
    """
    
    def __init__(self, 
                 shape: Optional[Union[int, Tuple[int, ...]]] = None,
                 dens: Optional[Union[int, Tuple[int, ...]]] = None,
                 center: Optional[Tuple[int, int]] = None,
                 radius: Optional[float] = None,
                 stride: Optional[float] = None,
                 padding: Optional[bool] = False,
                 base: Optional[Literal["horizontal", "vertical"]] = "horizontal",
                 tilt: Optional[Literal["forward", "backward", "inward", "outward"]] = "forward",
                 grid_type: Optional[Literal["rectangular", "triangular", "hexagonal", "voronoi", "spherical"]] = "rectangular",
                 octaves: Optional[int] = 0,
                 warp: Optional[float] = 0,
                 seed: Optional[int] = None):
        """
        Initialize a SurfaceGrid instance.
        
        Note
        ----
        For rectangular, triangular, and hexagonal grids shape is padded to three-integer tuple: 
        - the number of major grid nodes along x axis
        - the number of major grid nodes along y axis
        - the number of major grid nodes along t axis
        - all z coordinates are zeros
        For spherical grid shape is padded to two-integer tuple:
        - the number of icosahedron subdivisions
        - the number of major grid nodes along t axis
        
        Parameters
        ----------
        shape : Optional[Union[int, Tuple[int, ...]]]
            Shape of the major grid nodes. 
        dens : Optional[Union[int, Tuple[int, ...]]]
            Density of minor grid nodes.
        center : Optional[Tuple[int, int]]
            Center of the grid.
        radius : Optional[float]
            Radius of the sphere (used for "spherical" grid).
        stride : Optional[float]
            Stride of the grid (used for "rectangular", "triangular", and "hexagonal" grids).
        padding : Optional[bool], default False
            If True, pad the grid with one row and one column in each direction.
        base : Optional[Literal["horizontal", "vertical"]], default "horizontal"
            Base of the grid (used for "triangular" and "hexagonal" grids).
        tilt : Optional[Literal["forward", "backward", "inward", "outward"]], default "forward"
            Tilt of the grid (used for "triangular" and "hexagonal" grids).
        grid_type : Optional[Literal["rectangular", "triangular", "hexagonal", "voronoi", "spherical"]]
            Type of the grid to create.
        octaves : Optional[int], default 0
            Number of octaves for perlin noise.
        warp : Optional[float], default 0
            Warp factor for perlin noise.            
        seed : Optional[int], default None
            Seed for perlin noise.
            
        Raises
        ------
        ValueError: If both grid and other parameters are provided, or if grid is not 2D.
        """
        self._x = None
        self._y = None
        self._z = None
        self._noise = None
        self._type = None
        self._base = None
        self._tilt = None
        self._stride = None
        self._radius = None
        self._padding = False
        self._shape = None
        self._dens = None
        self._size = None
        self._octaves = None
        self._seed = None
        self._warp = None
        self._box = None
        self._tiling_box = None
        self._center = None
        self._centers = None
        self._faces = None
        self._is_padding = None
        self._triangles = None
        self._is_padding_triangles = None

        if shape is not None:
            # Check and validate parameters
            shape, dens, center, radius, stride, octaves, warp, seed = self._check_params(
                shape, dens, center, radius, stride, grid_type, octaves, warp, seed)
            if grid_type == "rectangular":
                self.make_rectangular_grid(shape, dens=dens, center=center, stride=stride, padding=padding, octaves=octaves, warp=warp, seed=seed)
            elif grid_type == "triangular":
                self.make_triangular_grid(shape, dens=dens, center=center, stride=stride, padding=padding, base=base, tilt=tilt, octaves=octaves, warp=warp, seed=seed)
            elif grid_type == "hexagonal":
                self.make_hexagonal_grid(shape, dens=dens, center=center, stride=stride, padding=padding, base=base, tilt=tilt, octaves=octaves, warp=warp, seed=seed)
            elif grid_type == "voronoi":
                self.make_voronoi_grid(shape, dens=dens, center=center, stride=stride, padding=padding, octaves=octaves, warp=warp, seed=seed)
            elif grid_type == "spherical":
                self.make_spherical_grid(shape, dens=dens, center=center, radius=radius, octaves=octaves, warp=warp, seed=seed)
        

    def _check_params(
        self,
        shape: Optional[Union[int, Tuple[int, ...]]] = None,
        dens: Optional[Union[int, Tuple[int, ...]]] = None,
        center: Optional[Tuple[float, ...]] = None,
        radius: Optional[float] = None,
        stride: Optional[float] = 1,
        grid_type: Optional[Literal["rectangular", "triangular", "hexagonal", "voronoi", "spherical"]] = "rectangular",
        octaves: Optional[int] = 0,
        warp: Optional[float] = 0,
        seed: Optional[int] = None,
    ) -> None:
        """
        Check and validate grid parameter ranges.
        
        Parameters
        ----------
        shape : Optional[Union[int, Tuple[int, ...]]]
            Shape of the grid
        dens : Optional[Union[int, Tuple[int, ...]]]
            Density of the grid
        center : Optional[Tuple[float, ...]]
            Center coordinates
        radius : Optional[float]
            Radius of the sphere
        stride : Optional[float]
            Grid stride
        grid_type : Optional[Literal["rectangular", "triangular", "hexagonal", "voronoi", "spherical"]]
            Type of grid to generate
        octaves : Optional[int]
            Number of octaves for Perlin noise
        warp : Optional[float]
            Warp factor for noise generation
        seed : Optional[int]
            Seed for random number generation
        """
        # Check shape
        if shape is not None:
            len_min = 1 if grid_type == "spherical" else 2
            len_max = 2 if grid_type == "spherical" else 3
            if isinstance(shape, int):
                new_shape = shape
                if shape < 1:
                    new_shape = 1
                    warnings.warn(f"Invalid shape: {shape}. Using {new_shape} instead.")
                shape = (new_shape,) * len_min
            elif len(shape) < len_min:
                new_shape = shape + (1,)
                warnings.warn(f"Invalid shape: {shape}. Using {new_shape} instead.")
                shape = new_shape
            elif len(shape) > len_max:
                warnings.warn(f"Invalid shape: {shape}. Using {shape[:len_max]} instead.")
                shape = shape[:len_max]
            elif any([s < 1 for s in shape]):
                new_shape = tuple([max(1, s) for s in shape])
                warnings.warn(f"Invalid shape: {shape}. Using {new_shape} instead.")
                shape = new_shape
        
        # Check density
        if dens is not None:
            length = len(shape)
            if isinstance(dens, int):
                new_dens = dens
                if dens < 1:
                    new_dens = max(1, dens)
                    warnings.warn(f"Invalid density: {dens}. Using {new_dens} instead.")
                dens = (new_dens,) * length
            elif len(dens) > length:
                warnings.warn(f"Invalid density: {dens}. Using {dens[:length]} instead.")
                dens = dens[:length]
            elif len(dens) < length:
                new_dens = dens + (1,) * (length - len(dens))
                warnings.warn(f"Invalid density: {dens}. Using {new_dens} instead.")
                dens = new_dens
            elif any([d < 1 for d in dens]):
                new_dens = tuple([max(1, d) for d in dens])
                warnings.warn(f"Invalid density: {dens}. Using {new_dens} instead.")
                dens = new_dens
        else:
            dens = (1,) * len(shape)

        # Check center
        if grid_type == "spherical":
            if center is None:
                center = (0.0, 0.0, 0.0)
            elif not isinstance(center, tuple) or len(center) != 3:
                warnings.warn(f"Invalid center for spherical grid: {center}. Using (0, 0, 0) instead.")
                center = (0.0, 0.0, 0.0)
        else:
            if center is None:
                center = (0.0,) * len(shape)
            elif not isinstance(center, tuple) or len(center) != len(shape):
                warnings.warn(f"Invalid center: {center}. Using (0, 0) instead.")
                center = (0.0,) * len(shape)

        # Check stride / radius
        if grid_type == "spherical":
            if stride is not None:
                warnings.warn(f"Stride is not used for spherical grid. Ignoring stride: {stride}.")
                stride = None
            if radius is None:
                radius = 1.0
            elif radius <= 0:
                warnings.warn(f"Invalid radius: {radius}. Using 1.0 instead.")
                radius = 1.0
        else:
            if radius is not None:
                warnings.warn(f"Radius is not used for non-spherical grid. Ignoring radius: {radius}.")
                radius = None
            if stride is None:
                stride = 1.0
            elif stride <= 0:
                warnings.warn(f"Invalid stride: {stride}. Using 1.0 instead.")
                stride = 1.0
                
        # Check Perlin noise parameters
        if octaves is not None and octaves < 0:
            warnings.warn(f"Invalid octaves: {octaves}. Using 0 instead.")
            octaves = 0
        
        if seed is not None and seed < 0:
            warnings.warn(f"Invalid seed: {seed}. Using 0 instead.")
            seed = 0
        
        if warp is not None and warp < 0.0:
            warnings.warn(f"Invalid warp: {warp}. Using 0.0 instead.")
            warp = 0.0
        
        return shape, dens, center, radius, stride, octaves, warp, seed


    def make_rectangular_grid(
        self,
        shape: Tuple[int, ...],
        dens: Union[int, Tuple[int, ...]] = 1,
        center: Optional[Tuple[int, int]] = None,
        stride: float = 1.0,
        padding: bool = False,
        octaves: int = 0,
        warp: float = 0,
        seed: int = None,
        nmax: int = 2000,
    ) -> None:
        """
        Generate a 2D rectangular grid.
        
        Parameters
        ----------
        shape : Tuple[int, ...]
            Shape of the major grid nodes.
        dens : int, default 1
            Number of grid nodes from one major node to the next along each axis. Positive integer.
        center : Optional[Tuple[int, int]], default None
            Center of the grid. If None, the center of the grid is the center of the shape.
        stride : float, default 1.0
            Stride of the grid.
        padding : bool, default False
            If True, pad the grid with one row and one column in each direction.
        octaves : int, default 0
            Number of octaves for perlin noise.
        warp : float, default 0
            Warp factor for perlin noise.
        seed : int, default None
            Seed for perlin noise.
        nmax : int, default 2000
            Maximum number of grid nodes in 2D plane to save resources while changing shape and dens.

        Raises
        ------
        Warning: If grid already exists
        ValueError: If shape and dens have different numbers of elements or are not 2D or 3D
        ValueError: If maximum grid nodes in 2D plane is exceeded
        """
        # Check and validate parameters
        radius = None
        shape, dens, center, radius, stride, octaves, warp, seed = self._check_params(
            shape, dens, center, radius, stride, "rectangular", octaves, warp, seed)

        # Convert shape and dens to tuples
        shape, dens = _shape_and_dens_to_tuples(shape, dens=dens)
        if len(shape) != len(dens) or len(shape) not in (2, 3):
            raise ValueError("Shape and dens must have the same number of elements and must be 2D or 3D.")
        
        if self._noise is not None:
            warnings.warn("Grid already exists. Skipping grid generation.")
            return
        
        # Grid type
        self._type = "rectangular"

        # Stride, x0, y0
        self._stride = stride
        size = tuple(s * d for s, d in zip(shape, dens))
        x0 = 0.5 * size[0] * stride
        y0 = 0.5 * size[1] * stride
        x0 = np.round(x0, 3)
        y0 = np.round(y0, 3)

        # Center
        if center is not None:
            self._center = center
        else:
            self._center = (x0, y0)

        # Bounding and tiling box
        self._box = ((self._center[0] - x0, self._center[1] - y0), (self._center[0] + x0, self._center[1] + y0))
        xmin, ymin = self._box[0]
        xmax, ymax = self._box[1]
        self._tiling_box = ((xmin, ymin), (xmax, ymax))

        # Calculate the size of the grid
        self._shape = shape
        self._dens = dens
        self._size = tuple((s + 2 * int(padding)) * d for s, d in zip(shape, dens))
        if np.prod(self._size[0:2]) > nmax:
            raise ValueError(f'Maximum grid nodes in 2D plane is {nmax}. Provided shape, dens, and padding produce {np.prod(self._size[0:2])} grid nodes in 2D plane.')

        # Create the grid
        x, y = [np.linspace(0, s, s + 1) for s in self._size[0:2]]
        x, y = np.meshgrid(x, y, indexing='ij')

        # Pad the grid
        self._padding = padding
        if padding:
            x -= self._dens[0]
            y -= self._dens[1]

        # Grid coordinates
        self._x = x * stride + self._center[0] - x0
        self._y = y * stride + self._center[1] - y0
        self._z = np.zeros_like(self._x)

        # Face indices
        self._faces = []
        self._is_padding = []
        self._triangles = []
        self._is_padding_triangles = []
        epsilon = 0.1 * self._stride
        for i in range(self._size[1]):
            for j in range(self._size[0]):
                idxs = [(i, j), (i, j+1), (i+1, j+1), (i+1, j)]
                x_, y_, z_ = self._get_face_center(face=idxs)
                is_padding = _outside_box(self._box, x_, y_, epsilon)
                self._faces.append(idxs)
                self._is_padding.append(is_padding)
                self._triangles.append([(i, j), (i+1, j+1), (i+1, j)])
                self._triangles.append([(i, j), (i, j+1), (i+1, j+1)])
                self._is_padding_triangles.append(is_padding)
                self._is_padding_triangles.append(is_padding)

        # Transpose face indices (y,x) -> (x,y)
        self._faces = _transpose_face_idxs(self._faces)
        self._triangles = _transpose_face_idxs(self._triangles)

        # Generate Perlin noise
        self.generate_noise(octaves, warp, seed)

        return


    def make_triangular_grid(
        self,
        shape: Tuple[int, ...],
        dens: Union[int, Tuple[int, ...]] = 1,
        center: Optional[Tuple[int, int]] = None,
        stride: float = 1.0,
        padding: bool = False,
        base: Literal["horizontal", "vertical"] = 'horizontal',
        tilt: Literal["forward", "backward", "inward", "outward"] = 'inward',
        octaves: int = 0,
        warp: float = 0,
        seed: int = None,
        nmax: int = 2000,
    ) -> None:
        """
        Make a triangular grid.

        Parameters
        ----------
        shape : Tuple[int, ...]
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
        octaves : int, default 0
            Number of octaves for perlin noise.
        warp : float, default 0
            Warp factor for perlin noise.
        seed : int, default None
            Seed for perlin noise.
        nmax : int, default 2000
            Maximum number of grid nodes in 2D plane to save resources while changing shape and dens.

        Raises
        ------
        ValueError: If shape and dens have different numbers of elements or are not 2D or 3D
        ValueError: If base is not "horizontal" or "vertical"
        ValueError: If tilt is not "forward", "backward", "inward", or "outward"
        ValueError: If maximum grid nodes in 2D plane is exceeded
        """
        # Check and validate parameters
        radius = None
        shape, dens, center, radius, stride, octaves, warp, seed = self._check_params(
            shape, dens, center, radius, stride, "triangular", octaves, warp, seed)

        # Convert shape and dens to tuples
        shape, dens = _shape_and_dens_to_tuples(shape, dens=dens)
        if len(shape) != len(dens) or len(shape) not in (2, 3):
            raise ValueError("Shape and dens must have the same number of elements and must be 2D or 3D.")
        
        # Check if base and tilt are valid
        if base not in ["horizontal", "vertical"]:
            raise ValueError("Base must be either 'horizontal' or 'vertical'.")
        if tilt not in ["forward", "backward", "inward", "outward"]:
            raise ValueError("Tilt must be either 'forward', 'backward', 'inward', or 'outward'.")
        
        # Check if grid already exists
        if self._noise is not None:
            warnings.warn("Grid already exists. Skipping grid generation.")
            return
        
        # Grid type
        self._type = "triangular"
        
        # Stride, x0, y0
        self._stride = stride
        size = tuple(s * d for s, d in zip(shape, dens))
        x0 = 0.5 * size[0] * stride
        y0 = 0.5 * size[1] * stride
        if base == "horizontal":
            y0 *= np.sqrt(3) / 2
            if tilt in ["forward", "backward"]:
                x0 += 0.25 * stride
        elif base == "vertical":
            x0 *= np.sqrt(3) / 2
            if tilt in ["forward", "backward"]:
                y0 += 0.25 * stride
        x0 = np.round(x0, 3)
        y0 = np.round(y0, 3)

        # Center
        if center is not None:
            self._center = center
        else:
            self._center = (x0, y0)

        # Bounding and tiling box
        self._box = ((self._center[0] - x0, self._center[1] - y0), (self._center[0] + x0, self._center[1] + y0))
        if tilt in ["forward", "backward"]:
            xmin, ymin = self._box[0]
            xmax, ymax = self._box[1]
            if base == "horizontal":
                xmax -= 0.5 * stride
            elif base == "vertical":
                ymax -= 0.5 * stride
            if base == "horizontal" and size[1] % 2 == 0:
                self._tiling_box = ((xmin, ymin), (xmax, ymax))
            if base == "vertical" and size[0] % 2 == 0:
                self._tiling_box = ((xmin, ymin), (xmax, ymax))

        # Calculate the size of the grid
        self._shape = shape
        self._dens = dens
        self._size = tuple((s + 2 * int(padding)) * d for s, d in zip(shape, dens))
        if np.prod(self._size[0:2]) > nmax:
            raise ValueError(f'Maximum grid nodes in 2D plane is {nmax}. Provided shape, dens, and padding produce {np.prod(self._size[0:2])} grid nodes in 2D plane.')

        # Create the grid
        x, y = [np.linspace(0, s, s + 1) for s in self._size[0:2]]
        x, y = np.meshgrid(x, y, indexing='ij')

        # Pad the grid
        tilt_ = tilt
        self._tilt = tilt
        self._base = base
        self._padding = padding
        if self._padding:
            x -= self._dens[0]
            y -= self._dens[1]
            if self._base == "horizontal" and self._dens[1] % 2 == 1:
                tilt_ = _reverse_tilt(tilt_)
            if self._base == "vertical" and self._dens[0] % 2 == 1:
                tilt_ = _reverse_tilt(tilt_)

        if self._base == 'horizontal':
            y *= np.sqrt(3) / 2
            x = _tilt_axis(x, tilt_)
            y[np.isnan(x)] = np.nan
        elif self._base == "vertical":
            x *= np.sqrt(3) / 2
            y = _tilt_axis(y.T, tilt_).T
            x[np.isnan(y)] = np.nan

        # Grid coordinates
        self._x = x * stride + self._center[0] - x0
        self._y = y * stride + self._center[1] - y0
        self._z = np.zeros_like(self._x)

        # Face indices
        self._faces = []
        self._is_padding = []
        epsilon = 0.1 * self._stride
        for i in range(self._size[1]):
            for j in range(self._size[0]):
                level = i + int(tilt_ in ["backward", "outward"])
                if self._base == "vertical":
                    level = j + int(tilt_ in ["backward", "outward"])
                if level % 2 == 0:
                    if np.isfinite(self._x[j+1, i+1]):
                        idxs = [(i, j), (i+1, j+1), (i+1, j)]
                        x_, y_, z_ = self._get_face_center(face=idxs)
                        self._faces.append(idxs)
                        self._is_padding.append(_outside_box(self._box, x_, y_, epsilon))
                        if np.isfinite(self._x[j+1, i]):
                            idxs = [(i, j), (i, j+1), (i+1, j+1)]
                            x_, y_, z_ = self._get_face_center(face=idxs)
                            self._faces.append(idxs)
                            self._is_padding.append(_outside_box(self._box, x_, y_, epsilon))
                else:
                    if np.isfinite(self._x[j+1, i]):
                        idxs = [(i, j), (i, j+1), (i+1, j)]
                        x_, y_, z_ = self._get_face_center(face=idxs)
                        self._faces.append(idxs)
                        self._is_padding.append(_outside_box(self._box, x_, y_, epsilon))
                        if np.isfinite(self._x[j+1, i+1]):
                            idxs = [(i+1, j), (i, j+1), (i+1, j+1)]
                            x_, y_, z_ = self._get_face_center(face=idxs)
                            self._faces.append(idxs)
                            self._is_padding.append(_outside_box(self._box, x_, y_, epsilon))

        # Triangulation indices
        self._triangles = self._faces
        self._is_padding_triangles = self._is_padding

        # Transpose face indices (y,x) -> (x,y)
        self._faces = _transpose_face_idxs(self._faces)
        self._triangles = _transpose_face_idxs(self._triangles)

        # Generate Perlin noise
        self.generate_noise(octaves, warp, seed)

        return


    def make_hexagonal_grid(
        self,
        shape: int,
        dens: int = 1,
        center: Optional[Tuple[float, float, float]] = None,
        stride: float = 1.0,
        padding: bool = False,
        base: Literal["horizontal", "vertical"] = "horizontal",
        tilt: Literal["forward", "backward", "inward", "outward"] = "forward",
        octaves: int = 0,
        warp: float = 0,
        seed: int = None,
        nmax: int = 2000,
    ) -> None:
        """
        Make a hexagonal grid.

        Parameters
        ----------
        shape : int
            The shape of the grid.
        dens : int, default 1
            The density of the grid.
        center : Optional[Tuple[float, float, float]], default None
            The center of the grid.
        stride : float, default 1.0 
            The stride of the grid.
        padding : bool, default False
            Whether to pad the grid.
        base : Literal["horizontal", "vertical"], default "horizontal"
            The base of the grid.
        tilt : Literal["forward", "backward", "inward", "outward"], default "forward"
            The tilt of the grid.
        octaves : int, default 0
            Number of octaves for perlin noise.
        warp : float, default 0
            Warp factor for perlin noise.
        seed : int, default None
            Seed for perlin noise.
        nmax : int, default 2000
            Maximum number of grid nodes in 2D plane to save resources while changing shape and dens.

        Raises
        ------
        ValueError: If shape and dens have different numbers of elements or are not 2D or 3D
        ValueError: If base is not "horizontal" or "vertical"
        ValueError: If tilt is not "forward", "backward", "inward", or "outward"
        ValueError: If maximum grid nodes in 2D plane is exceeded
        """
        # Check and validate parameters
        radius = None
        shape, dens, center, radius, stride, octaves, warp, seed = self._check_params(
            shape, dens, center, radius, stride, "hexagonal", octaves, warp, seed)

        # Convert shape and dens to tuples
        shape, dens = _shape_and_dens_to_tuples(shape, dens=dens)
        if len(shape) != len(dens) or len(shape) not in (2, 3):
            raise ValueError("Shape and dens must have the same number of elements and must be 2D or 3D.")  
        
        # Check if base and tilt are valid
        if base not in ["horizontal", "vertical"]:
            raise ValueError("Base must be either 'horizontal' or 'vertical'.")
        if tilt not in ["forward", "backward", "inward", "outward"]:
            raise ValueError("Tilt must be either 'forward', 'backward', 'inward', or 'outward'.")
        
        # Check if grid already exists
        if self._noise is not None:
            warnings.warn("Grid already exists. Skipping grid generation.")
            return
        
        # Grid type
        self._type = "hexagonal"

        # Calculate parameters of triangular grid
        self._stride = stride
        size = tuple(s * d for s, d in zip(shape, dens))
        size_triang, base_triang, tilt_triang = _hex_to_triang_size(size, base, tilt)

        # Stride
        x0 = 0.5 * size_triang[0] * stride
        y0 = 0.5 * size_triang[1] * stride
        if base_triang == "horizontal":
            y0 *= np.sqrt(3) / 2
            if tilt_triang in ["forward", "backward"]:
                x0 += 0.25 * stride  
        elif base_triang == "vertical":
            x0 *= np.sqrt(3) / 2
            if tilt_triang in ["forward", "backward"]:
                y0 += 0.25 * stride
        x0 = np.round(x0, 3)
        y0 = np.round(y0, 3)

        # Center
        if center is not None:
            self._center = center
        else:
            self._center = (x0, y0)

        # Bounding and tiling box
        self._box = ((self._center[0] - x0, self._center[1] - y0), (self._center[0] + x0, self._center[1] + y0))
        if tilt in ["forward", "backward"] and tilt_triang in ["forward", "backward"]:
            xmin, ymin = self._box[0]
            xmax, ymax = self._box[1]
            if base_triang == "horizontal":
                xmax -= 0.5 * stride
                ymax -= 0.5 * stride * np.sqrt(3)
            elif base_triang == "vertical":
                xmax -= 0.5 * stride * np.sqrt(3)
                ymax -= 0.5 * stride
            xmax = np.round(xmax, 3)
            ymax = np.round(ymax, 3)
            self._tiling_box = ((xmin, ymin), (xmax, ymax))

        # Calculate the size of the grid
        self._shape = shape
        self._dens = dens
        self._size = tuple((s + 2 * int(padding)) * d for s, d in zip(shape, dens))
        if np.prod(self._size[0:2]) > nmax:
            raise ValueError(f'Maximum grid nodes in 2D plane is {nmax}. Provided shape, dens, and padding produce {np.prod(self._size[0:2])} grid nodes in 2D plane.')

        # Tilt hexagonal grid
        tilt_ = tilt
        self._tilt = tilt
        self._base = base
        self._padding = padding
        if self._padding:
            if self._base == "horizontal" and self._dens[0] % 2 == 1:
                tilt_ = _reverse_tilt(tilt_)
            if self._base == "vertical" and self._dens[1] % 2 == 1:
                tilt_ = _reverse_tilt(tilt_)


        # Create the grid
        size_triang, base_triang, tilt_triang = _hex_to_triang_size(self._size, base, tilt)
        x, y = [np.linspace(0, s, s + 1) for s in size_triang[0:2]]
        x, y = np.meshgrid(x, y, indexing='ij')

        # Pad the grid
        if padding:
            xscale = 3 / 2 if base == "horizontal" else 2
            yscale = 2 if base == "horizontal" else 3 / 2
            x -= dens[0] * xscale
            y -= dens[1] * yscale
            if base_triang == "horizontal" and dens[0] % 2 == 1:
                tilt_triang = _reverse_tilt(tilt_triang)
            if base_triang == "vertical" and dens[1] % 2 == 1:
                tilt_triang = _reverse_tilt(tilt_triang)

        if base_triang == "horizontal":
            y *= np.sqrt(3) / 2
            x = _tilt_axis(x, tilt_triang)
            y[np.isnan(x)] = np.nan
        elif base_triang == "vertical":
            x *= np.sqrt(3) / 2
            y = _tilt_axis(y.T, tilt_triang).T
            x[np.isnan(y)] = np.nan

        # Set nan values
        x = _hex_to_triang_nan(x, tilt_, base)
        y = _hex_to_triang_nan(y, tilt_, base)

        # Grid coordinates
        self._x = x * stride + self._center[0] - x0
        self._y = y * stride + self._center[1] - y0
        self._z = np.zeros_like(self._x)

        # Face centers
        self._centers = []
        if self._base == "horizontal":
            if tilt_ in ["forward", "outward"]:
                for i in range(1, size_triang[1]):
                    if i % 2 == 0:
                        for j in range(size_triang[0])[2::3]:
                            self._centers.append((i,j))
                    if i % 2 == 1:
                        for j in range(size_triang[0])[1::3]:
                            self._centers.append((i,j))
            else:
                for i in range(1, size_triang[1]):
                    if i % 2 == 0:
                        for j in range(size_triang[0])[1::3]:
                            self._centers.append((i,j))
                    if i % 2 == 1:
                        for j in range(size_triang[0])[2::3]:
                            self._centers.append((i,j))
        elif self._base == "vertical":
            if tilt_ in ["forward", "outward"]:
                for j in range(1, size_triang[0]):
                    if j % 2 == 0:
                        for i in range(size_triang[1])[2::3]:
                            self._centers.append((i,j))
                    if j % 2 == 1:
                        for i in range(size_triang[1])[1::3]:
                            self._centers.append((i,j))
            else:
                for j in range(1, size_triang[0]):
                    if j % 2 == 0:
                        for i in range(size_triang[1])[1::3]:
                            self._centers.append((i,j))
                    if j % 2 == 1:
                        for i in range(size_triang[1])[2::3]:
                            self._centers.append((i,j))

        # Face padding
        self._is_padding = []
        epsilon = 0.1 * stride
        for i, j in self._centers:
            x_ = self._x[j, i]
            y_ = self._y[j, i]
            is_padding = _outside_box(self._box, x_, y_, epsilon)
            self._is_padding.append(is_padding)

        # Face and triangulation indices
        self._faces = []
        self._triangles = []
        self._is_padding_triangles = []
        for k, (i, j) in enumerate(self._centers):
            if self._base == "horizontal":
                dj = 0
                if tilt_ in ["forward", "outward"] and i % 2 == 0:
                    dj = 1
                if tilt_ in ["backward", "inward"] and i % 2 == 1:
                    dj = 1
                idxs = [(i, j+1), (i+1, j+dj), (i+1, j+dj-1), (i, j-1), (i-1, j+dj-1), (i-1, j+dj), (i, j+1)]
                self._faces.append(idxs[:-1])
                for q, idx in enumerate(idxs[:-1]):
                    triangle = [idx, idxs[q+1], (i,j)]
                    self._triangles.append(triangle)
                    self._is_padding_triangles.append(self._is_padding[k])
            elif self._base == "vertical":
                di = 0
                if tilt_ in ["forward", "outward"] and j % 2 == 0:
                    di = 1
                if tilt_ in ["backward", "inward"] and j % 2 == 1:
                    di = 1
                idxs = [(i+1, j), (i+di, j+1), (i+di-1, j+1), (i-1, j), (i+di-1, j-1), (i+di, j-1), (i+1, j)]
                self._faces.append(idxs[:-1])
                for q, idx in enumerate(idxs[:-1]):
                    triangle = [idx, idxs[q+1], (i,j)]
                    self._triangles.append(triangle)
                    self._is_padding_triangles.append(self._is_padding[k])

        # Transpose face indices (y,x) -> (x,y)
        self._faces = _transpose_face_idxs(self._faces)
        self._centers = [(j, i) for i, j in self._centers]
        self._triangles = _transpose_face_idxs(self._triangles)

        # Generate Perlin noise
        self.generate_noise(octaves, warp, seed)

        return


    def make_voronoi_grid(
        self,
        shape: Tuple[int, ...],
        dens: Union[int, Tuple[int, ...]] = 1,
        center: Optional[Tuple[int, int]] = None,
        stride: float = 1.0,
        padding: bool = False,
        octaves: int = 0,
        warp: float = 0,
        seed: int = None,
        nmax: int = 2000,
    ) -> None:
        """
        Generate a 2D rectangular grid.
        
        Parameters
        ----------
        shape : Tuple[int, ...]
            Shape of the major grid nodes.
        dens : int, default 1
            Number of grid nodes from one major node to the next along each axis. Positive integer.
        center : Optional[Tuple[int, int]], default None
            Center of the grid. If None, the center of the grid is the center of the shape.
        stride : float, default 1.0
            Stride of the grid.
        padding : bool, default False
            If True, pad the grid with one row and one column in each direction.
        octaves : int, default 0
            Number of octaves for perlin noise.
        warp : float, default 0
            Warp factor for perlin noise.
        seed : int, default None
            Seed for perlin noise.
        nmax : int, default 2000
            Maximum number of grid nodes in 2D plane to save resources while changing shape and dens.

        Raises
        ------
        Warning: If grid already exists
        ValueError: If shape and dens have different numbers of elements or are not 2D or 3D
        ValueError: If maximum grid nodes in 2D plane is exceeded
        """
        # Check and validate parameters
        radius = None
        shape, dens, center, radius, stride, octaves, warp, seed = self._check_params(
            shape, dens, center, radius, stride, "voronoi", octaves, warp, seed)

        # Convert shape and dens to tuples
        shape, dens = _shape_and_dens_to_tuples(shape, dens=dens)
        if len(shape) != len(dens) or len(shape) not in (2, 3):
            raise ValueError("Shape and dens must have the same number of elements and must be 2D or 3D.")
        
        if self._noise is not None:
            warnings.warn("Grid already exists. Skipping grid generation.")
            return
        
        # Assign grid type
        self._type = "voronoi"

        # Stride, x0, y0
        self._stride = stride
        size = tuple(s * d for s, d in zip(shape, dens))
        x0 = 0.5 * size[0] * stride
        y0 = 0.5 * size[1] * stride
        x0 = np.round(x0, 3)
        y0 = np.round(y0, 3)

        # Center
        if center is not None:
            self._center = center
        else:
            self._center = (x0, y0)

        # Bounding and tiling box
        self._box = ((self._center[0] - x0, self._center[1] - y0), (self._center[0] + x0, self._center[1] + y0))
        xmin, ymin = self._box[0]
        xmax, ymax = self._box[1]
        self._tiling_box = ((xmin, ymin), (xmax, ymax))

        # Calculate the size of the grid
        self._shape = shape
        self._dens = dens
        self._size = tuple((s + 2 * int(padding)) * d for s, d in zip(shape, dens))
        if np.prod(self._size[0:2]) > nmax:
            raise ValueError(f'Maximum grid nodes in 2D plane is {nmax}. Provided shape, dens, and padding produce {np.prod(self._size[0:2])} grid nodes in 2D plane.')

        # Create the grid for cell centers and move coordinates to the center of the cell
        x, y = [np.linspace(0, s, s + 1)[:-1] + 0.5 for s in self._size[0:2]]
        x, y = np.meshgrid(x, y, indexing='ij')

        # Pad the grid
        self._padding = padding
        if padding:
            x -= self._dens[0]
            y -= self._dens[1]

        # Grid coordinates
        self._x = x * stride + self._center[0] - x0
        self._y = y * stride + self._center[1] - y0
        self._z = np.zeros_like(self._x)

        # Face centers
        self._centers = []
        self._is_padding = []
        epsilon = 0.1 * self._stride
        for i in range(self._size[0]):
            for j in range(self._size[1]):
                self._centers.append((i,j))
                x_ = self._x[i,j]
                y_ = self._y[i,j]
                is_padding = _outside_box(self._box, x_, y_, epsilon)
                self._is_padding.append(is_padding)

        # Generate Perlin noise
        self.generate_noise(octaves, warp, seed)
        
        return    

    def make_spherical_grid(
        self,
        shape: Union[int, Tuple[int, ...]],
        dens: Union[int, Tuple[int, ...]] = 1,
        center: Optional[Tuple[float, float, float]] = None,
        radius: float = 1.0,
        octaves: int = 0,
        warp: float = 0,
        seed: int = None,
        nmax: int = 6,
    ) -> None:
        """Make a spherical grid based on subdivided icosahedron.
        
        Parameters
        ----------
        shape : Union[int, Tuple[int, int]]
            The number of subdivisions of the icosahedron. And, if tuple, the number of noise layers.
        dens : Union[int, Tuple[int, ...]], default 1
            The density of the noise grid. 
        center : Optional[Tuple[float, float, float]], default None
            The center of the sphere.
        radius : float, default 1.0
            Radius of the sphere.
        octaves : int, default 0
            Number of octaves for perlin noise.
        warp : float, default 0
            Warp factor for perlin noise.
        seed : int, default None
            Seed for perlin noise.
        nmax : int, default 6
            Maximum number of icosahedron subdivisions to save resources while changing shape and dens.

        Raises
        ------
        Value Error: If shape is not a positive integer.
        Value Error: If density is not a tuple of positive integers.
        """
        if isinstance(shape, int):
            shape = (shape,)
        if isinstance(dens, int):
            dens = (dens,) * len(shape)
        elif len(dens) != len(shape):
            raise ValueError(f"Invalid density: {dens}. Must be a tuple of the same length as shape.")

        # Check and validate parameters
        stride = None
        shape, dens, center, radius, stride, octaves, warp, seed = self._check_params(
            shape, dens, center, radius, stride, "spherical", octaves, warp, seed)

        # Check if grid already exists
        if self._noise is not None:
            warnings.warn("Grid already exists. Skipping grid generation.")
            return
        
        # Assign grid type
        self._type = "spherical"
        
        # Generate grid
        self._shape = shape
        self._dens = dens
        self._center = center
        self._radius = radius
        self._box = ((0, 0), (2 * np.pi, np.pi))
        self._tiling_box = ((0, 0), (2 * np.pi, np.pi))
        self._r, self._theta, self._phi, self._faces = _make_icosahedron(shape[0], nmax)
        self._x = self._radius * np.sin(self._theta) * np.cos(self._phi) + self._center[0]
        self._y = self._radius * np.sin(self._theta) * np.sin(self._phi) + self._center[1]
        self._z = self._radius * np.cos(self._theta) + self._center[2]
        self._r = self._radius * self._r
        self._is_padding = [False] * len(self._faces)
        self._triangles = self._faces
        self._is_padding_triangles = self._is_padding

        # Generate Perlin noise
        self.generate_noise(octaves, warp, seed)

        return


    def generate_noise(
        self,
        octaves: int = 0,
        warp: float = 0,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate noise based on the grid type.

        Parameters
        ----------
        octaves : int, default 0
            Number of octaves for perlin noise.
        warp : float, default 0
            Warp factor for perlin noise.
        seed : Optional[int], default None
            Seed for perlin noise.

        Returns
        -------
        np.ndarray
            Generated noise.
        """
        # Set parameters
        self._octaves = octaves
        self._warp = warp
        self._seed = seed

        # Rectangular grid
        if self._type == "rectangular":
            # Generate noise
            noise = perlin(self._shape, dens=self._dens, octaves=self._octaves, seed=self._seed, warp=self._warp)
            # Pad noise along x axis
            i1 = self._dens[0] if self._padding else 0
            i2 = self._dens[0] + 1 if self._padding else 1
            noise = _tile_noise(noise, i1, i2, axis=0)
            # Pad noise along y axis
            i1 = self._dens[1] if self._padding else 0
            i2 = self._dens[1] + 1 if self._padding else 1
            noise = _tile_noise(noise, i1, i2, axis=1)
            # Save noise as attribute
            self._noise = noise

        # Triangular grid
        elif self._type == "triangular":
            # Generate noise
            noise = perlin(self._shape, dens=self._dens, octaves=self._octaves, seed=self._seed, warp=self._warp)
            # Pad noise along x axis
            i1 = self._dens[0] if self._padding else 0
            i2 = self._dens[0] + 1 if self._padding else 1
            noise = _tile_noise(noise, i1, i2, axis=0)
            # Pad noise along y axis
            i1 = self._dens[1] if self._padding else 0
            i2 = self._dens[1] + 1 if self._padding else 1
            noise = _tile_noise(noise, i1, i2, axis=1)
            # Save noise as attribute
            self._noise = noise

        # Hexagonal grid
        elif self._type == "hexagonal":
            dens = self._dens
            shape = self._shape
            # Generate noise
            pad_mask = _create_padding_mask(self._x, self._triangles, self._is_padding_triangles)
            if np.any(pad_mask):
                i0, i1 = np.where(pad_mask)[0].min(), np.where(pad_mask)[0].max()
                j0, j1 = np.where(pad_mask)[1].min(), np.where(pad_mask)[1].max()
                noise_size = [i1 - i0, j1 - j0]
                if self._base == "horizontal":
                    noise_size[1] = max(2, noise_size[1] - 1)
                    if self._padding and shape[0] % 2 == 0  and dens[0] % 2 == 1:
                        noise_size[0] = max(2, noise_size[0] - 1)
                elif self._base == "vertical":
                    noise_size[0] = max(2, noise_size[0] - 1)
                    if self._padding and shape[1] % 2 == 0  and dens[1] % 2 == 1:
                        noise_size[1] = max(2, noise_size[1] - 1)
                tile = self._tiling_box is not None
                noise_shape, noise_dens = _hex_to_noise_shape_and_dens(noise_size, self._shape, self._dens, tile)
                noise = perlin(noise_shape, dens=noise_dens, octaves=self._octaves, seed=self._seed, warp=self._warp)
                noise = noise[:noise_size[0]]
                noise = noise[:,:noise_size[1]]
                # Pad noise along x axis
                i1 = np.where(pad_mask)[0].min() if np.any(pad_mask) else 0
                i2 = self._x.shape[0] - i1 - noise.shape[0]
                noise = _tile_noise(noise, i1, i2, axis=0)
                # Pad noise along y axis
                i1 = np.where(pad_mask)[1].min() if np.any(pad_mask) else 0
                i2 = self._x.shape[1] - i1 - noise.shape[1]
                noise = _tile_noise(noise, i1, i2, axis=1)
            else:
                noise = np.zeros(self._x.shape + (self._shape[-1] * self._dens[-1],))
            # Save noise as attribute
            self._noise = noise

        # Voronoi grid
        elif self._type == "voronoi":
            # Generate noise
            noise = perlin(self._shape, dens=self._dens, octaves=octaves, seed=seed, warp=warp)
            # Pad noise along x axis
            i1 = self._dens[0] if self._padding else 0
            i2 = self._dens[0] if self._padding else 0
            noise = _tile_noise(noise, i1, i2, axis=0)
            # Pad noise along y axis
            i1 = self._dens[1] if self._padding else 0
            i2 = self._dens[1] if self._padding else 0
            noise = _tile_noise(noise, i1, i2, axis=1)
            # Save noise as attribute
            self._noise = noise

        # Spherical grid
        elif self._type == "spherical":
            # Adjust grid shape and density
            dens = self._dens
            shape = self._shape
            shape0, dens0 = int(np.ceil(self._x.shape[0] / dens[0])), dens[0]
            size1 = self._x.shape[1] - 1
            shape1, dens1 = size1, 1
            if dens[0] > 1 and size1 > 5:
                denses = np.array([d for d in range(1,size1) if size1 % d == 0])
                i = np.argmin(np.abs(denses - dens[0]))
                dens1 = denses[i]
                shape1 = size1 // dens1
            if len(shape) == 1:
                grid_shape = (shape0, shape1)
                grid_dens = (dens0, dens1)
            else:
                grid_shape = (shape0, shape1, shape[1])
                grid_dens = (dens0, dens1, dens[1])
            # Generate noise
            noise = perlin(grid_shape, dens=grid_dens, octaves=octaves, seed=seed, warp=warp)
            noise = noise if noise.ndim == 3 else noise[..., np.newaxis]
            # Truncate and pad the noise grid
            noise = np.concatenate([noise, noise[:,:1]], axis=1)
            noise = noise[:self._x.shape[0]]
            # Smooth noise along latitude and longitude
            grid = []
            n_max = self._x.shape[1]
            n_points = np.sum(~np.isnan(self._x), axis=1)
            kernel = _spherical_kernels(n_points, grid_dens[1], scale=1)
            window = np.array([len(k) for k in kernel])
            for i in range(noise.shape[2]):
                slice = noise[...,i]
                # Latitude smoothing
                smoothed = []
                for j in range(slice.shape[0]):
                    y = np.zeros((n_max,)) * np.nan
                    if n_points[j] == n_max:
                        smoothed.append(slice[j])
                    elif n_points[j] < 2:
                        y = slice[j]
                        y[1:] = np.nan
                        smoothed.append(y)
                    else:
                        x = np.linspace(0, 1, n_points[j])
                        xp = np.linspace(0, 1, n_max)
                        y = np.interp(x, xp, slice[j])
                        if len(kernel[j]) > 1:
                            y = _latitude_smoothing(y, kernel[j])
                        y_ = np.zeros((n_max,)) * np.nan
                        y_[:n_points[j]] = y
                        smoothed.append(y_)
                smoothed = np.stack(smoothed)
                if window[1] > 1:
                    smoothed[0,0] = np.nanmean(smoothed[1])
                    smoothed[-1,0] = np.nanmean(smoothed[-2])
                # Longitude smoothing
                if grid_dens[1] > 1:
                    smoothed = _longitude_smoothing(smoothed, n_points, kernel)
                grid.append(smoothed)
            # Save noise as attribute
            noise = np.stack(grid, axis=2)
            if noise.shape[2] == 1 and len(self._shape) == 1:
                noise = noise[..., 0]
            self._noise = noise
            self._size = self._noise.shape

        return np.copy(self._noise)


    def apply_noise(self, mode: Literal["xy", "z"] = "xy", scale=0.25) -> np.ndarray:
        """
        Apply noise to the grid.

        Note
        ----
        Changes the coordinates attributes!

        Parameters
        ----------
        mode : Literal["xy", "z"], default "xy"
            Whether to apply noise to (x,y) or to z.
        scale : float, default 0.25
            Scale of the noise.

        Returns
        -------
        np.ndarray
            X coordinates of the grid.
        np.ndarray
            Y coordinates of the grid.
        np.ndarray
            Z coordinates of the grid.
        """
        # Scale noise
        noise = np.copy(self._noise)
        if noise.ndim > 2:
            noise = noise[..., 0]
        # Spherical grid
        if self._type == "spherical":
            r, theta, phi = self.get_coords(mode="spherical")
            # Decompose noise into phi and theta components using Euler's formula
            if mode == "xy":
                z = np.exp(2j * np.pi * noise)
                phi = phi + scale * self._radius * z.real
                theta = theta + scale * self._radius * z.imag
                x, y, z = spherical_to_cartesian(r, theta, phi)
                self._x = x
                self._y = y
                self._z = z
                self._phi = phi
                self._theta = theta
            # Or add noise to r
            elif mode == "z":
                r = r + scale * self._radius * noise / np.std(noise)
                x, y, z = spherical_to_cartesian(r, theta, phi)
                self._x = x
                self._y = y
                self._z = z
                self._r = r
        # Other grid types
        else:
            # Decompose noise into x and y components using Euler's formula
            if mode == "xy":
                z = np.exp(2j * np.pi * noise)
                self._x = self._x + scale * self._stride * z.real
                self._y = self._y + scale * self._stride * z.imag
            # Or add noise to z
            elif mode == "z":
                self._z = self._z + scale * self._stride * noise / np.std(noise)
        return self._x, self._y, self._z


    def _voronoi_coords_and_padding(
        self,
        face_type: Literal["faces", "triangles"] = "faces",
        x: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        z: Optional[np.ndarray] = None,
    ) -> Tuple[List[List[Tuple[float, float, float]]], List[bool]]:
        """
        Calculate face coordinates for cellular noise using Voronoi tessellation.
        
        Parameters
        ----------
        face_type : Literal["faces", "triangles"]
            Whether to return faces or triangles.
        x : np.ndarray
            X coordinates of cell centers
        y : np.ndarray
            Y coordinates of cell centers
        z : np.ndarray
            Z coordinates of cell centers
            
        Returns
        -------
        List[List[Tuple[float, float, float]]]
            List of faces, where each face is a list of (x,y,z) vertex coordinates.
        List[bool]
            List of booleans indicating whether each face is a padding face.

        """
        # Get coordinates
        x = x if x is not None else self._x
        y = y if y is not None else self._y
        z = z if z is not None else self._z

        # Tiling dx, dy
        dx = self._tiling_box[1][0] - self._tiling_box[0][0]
        dy = self._tiling_box[1][1] - self._tiling_box[0][1]
        if self._padding:
            dx += 2 * self._dens[0] * self._stride
            dy += 2 * self._dens[1] * self._stride

        # Make superpadding points
        x_ = np.concatenate([x[:,-2:], x, x[:,:2]], axis=1)
        y_ = np.concatenate([y[:,-2:] - dy, y, y[:,:2] + dy], axis=1)
        z_ = np.concatenate([z[:,-2:], z, z[:,:2]], axis=1)
        x_ = np.concatenate([x_[-2:] - dx, x_, x_[:2] + dx], axis=0)
        y_ = np.concatenate([y_[-2:], y_, y_[:2]], axis=0)
        z_ = np.concatenate([z_[-2:], z_, z_[:2]], axis=0)
        
        # Make superpadding mapping indices
        # (superpadding points have index = -1)
        n_faces = x.shape[0] * x.shape[1]
        idxs = np.arange(n_faces, dtype=int)
        idxs = idxs.reshape(x.shape)
        pad0 = -np.ones((x.shape[0], 2), dtype=int)
        pad1 = -np.ones((2, x.shape[1] + 4), dtype=int)
        idxs = np.concatenate([pad0, idxs, pad0], axis=1)
        idxs = np.concatenate([pad1, idxs, pad1], axis=0)

        # Flatten idxs, x_, y_, z_
        idxs = idxs.flatten()
        x_ = x_.flatten()
        y_ = y_.flatten()
        z_ = z_.flatten()

        # Points for 2D Voronoi tessellation
        points = np.stack([x_, y_], axis=1)

        # Voronoi tessellation
        vor = Voronoi(points)

        # Create mapping of point idxs
        face_idxs = {}
        for j, k in enumerate(idxs):
            if k >= 0:
                face_idxs[k] = j

        # Create mapping of vertex indices to their adjacent regions
        vertex_regions = {}
        for j, region in enumerate(vor.regions):
            for i in region:
                if i not in vertex_regions:
                    vertex_regions[i] = []
                vertex_regions[i].append(j)

        # Create mapping of point to region and backwards
        point_to_region = {i: vor.point_region[i] for i in range(len(points))}
        region_to_point = {j: i for i, j in point_to_region.items()}

        # Calculate z for each vertex
        vertex_z = {}
        for i in vertex_regions:
            vertex_points = [region_to_point[j] for j in vertex_regions[i]]
            vertex_z[i] = np.mean([z_[k] for k in vertex_points])

        # Collect face coordinates
        face_coords = []
        face_padding = self._is_padding

        # Process only original points (excluding superpadding)
        for k in range(n_faces):
            j = face_idxs[k]
            # Get region index for current point
            region_idx = vor.point_region[j]
            # Get vertex indices for this region
            vertex_idxs = vor.regions[region_idx]

            # Get vertex coordinates
            vertices = []
            for i in vertex_idxs:
                vertex = vor.vertices[i]
                # Add interpolated z-coordinate
                vertices.append((float(vertex[0]), float(vertex[1]), float(vertex_z[i])))

            # Sort vertices counterclockwise around cell center
            center_point = np.array([points[j][0], points[j][1]])
            vertices = sorted(vertices,
                key=lambda v: np.arctan2(v[1] - center_point[1], v[0] - center_point[0]))
            
            face_coords.append(vertices)

        # Triangulation
        centers = self._centers
        if face_type == "triangles":
            triangle_coords = []
            triangle_padding = []
            for k, center in enumerate(centers):
                i, j = center
                x0 = (x[i,j], y[i,j], z[i,j])
                vertices = face_coords[k]
                # Create triangles by connecting center to each pair of consecutive vertices
                for i in range(len(vertices)):
                    x1 = vertices[i]
                    x2 = vertices[(i + 1) % len(vertices)]
                    triangle_coords.append([x0, x1, x2])
                    triangle_padding.append(face_padding[k])
            face_coords = triangle_coords
            face_padding = triangle_padding

        return face_coords, face_padding


    def _get_face_coords_and_padding(
        self,
        face_type: Literal["faces", "triangles"] = "faces",
        x: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        z: Optional[np.ndarray] = None,
    ) -> Tuple[List[List[Tuple[float, float, float]]], List[bool]]:
        """
        Get face coordinates and padding for a given face type.
        """
        if self._type == "voronoi":
            face_coords, face_padding = self._voronoi_coords_and_padding(face_type, x, y, z)
        else:
            # Get coordinates
            x = x if x is not None else self._x
            y = y if y is not None else self._y
            z = z if z is not None else self._z

            # For other grid types, use existing face indices
            face_coords = []
            face_idxs = self.get_face_idxs(face_type)
            for face in face_idxs:
                coords = []
                for i, j in face:
                    coords.append((
                        float(x[i,j]),
                        float(y[i,j]),
                        float(z[i,j]) if z is not None else 0.0
                    ))
                face_coords.append(coords)
            face_padding = self._is_padding if face_type == "faces" else self._is_padding_triangles

        return face_coords, face_padding


    def get_face_coords(
        self, 
        face_type: Literal["faces", "triangles"] = "faces", 
        x: Optional[np.ndarray] = None, 
        y: Optional[np.ndarray] = None,
        z: Optional[np.ndarray] = None,
    ) -> List[List[Tuple[float, float, float]]]:
        """Get coordinates of faces based on their indices.
        
        Parameters
        ----------
        face_type : Literal["faces", "triangles"], default "faces"
            Whether to return faces or triangles.
        x : Optional[np.ndarray], default None
            X coordinates. If None, uses self._x.
        y : Optional[np.ndarray], default None
            Y coordinates. If None, uses self._y.
        z : Optional[np.ndarray], default None
            Z coordinates. If None, uses self._z.

        Returns
        -------
        List[List[Tuple[float, float, float]]]
            List of face coordinates, where each face is a list of (x,y,z) coordinate tuples.
        """
        face_coords, _ = self._get_face_coords_and_padding(face_type, x, y, z)
        return face_coords


    def get_face_idxs(
        self, 
        face_type: Literal["faces", "triangles"] = "faces"
    ) -> List[List[Tuple[int, int]]]:
        """
        Get the indices of the faces of the grid.

        Parameters
        ----------
        face_type : Literal["faces", "triangles"], default "faces"
            Whether to return faces or triangles.

        Returns
        -------
        List[List[Tuple[int, int]]]
            List of face indices, where each face is a list of (i,j) index tuples.
        """
        if self._type == "voronoi":
            return None
        idxs = []
        if face_type == "triangles":
            idxs = [face for face in self._triangles]
        else:
            idxs = [face for face in self._faces]
        return idxs


    def get_face_normals(
        self,
        face_type: Literal["faces", "triangles"] = "faces",
        x: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        z: Optional[np.ndarray] = None
    ) -> List[Tuple[float, float, float]]:
        """
        Get the normals of the faces of the grid.

        Parameters
        ----------
        face_type : Literal["faces", "triangles"], default "faces"
            Whether to return normals of faces or triangles.
        x : Optional[np.ndarray], default None
            X coordinates. If None, uses self._x.
        y : Optional[np.ndarray], default None
            Y coordinates. If None, uses self._y.
        z : Optional[np.ndarray], default None
            Z coordinates. If None, uses self._z.

        Returns
        -------
        List[Tuple[float, float, float]]
            List of face normals, where each normal is a (x,y,z) coordinate tuple.
        """
        # Get face coordinates
        face_coords, _ = self._get_face_coords_and_padding(face_type, x, y, z)

        normals = []
        for coords in face_coords:
            # Get vertices
            v0 = np.asarray(coords[0])
            v1 = np.asarray(coords[1])
            v2 = np.asarray(coords[2])

            # Calculate normal
            edge1 = v1 - v0
            edge2 = v2 - v1

            normal = np.cross(edge1, edge2)
            normal = normal / np.linalg.norm(normal)
            normals.append(normal)

        return normals


    def get_face_centers(
        self, 
        face_type: Literal["faces", "triangles"] = "faces", 
        x: Optional[np.ndarray] = None, 
        y: Optional[np.ndarray] = None,
        z: Optional[np.ndarray] = None
    ) -> List[Tuple[float, float, float]]:
        """
        Get center coordinates of faces based on their indices.
        
        Parameters
        ----------
        face_type : Literal["faces", "triangles"], default "faces"
            Whether to return centers of faces or triangles.
        x : Optional[np.ndarray], default None
            X coordinates. If None, uses self._x.
        y : Optional[np.ndarray], default None
            Y coordinates. If None, uses self._y.
        z : Optional[np.ndarray], default None
            Z coordinates. If None, uses self._z.

        Returns
        -------
        List[Tuple[float, float, float]]
            List of face center coordinates, where each center is a (x,y,z) coordinate tuple.
            
        Raises
        ------
        ValueError: If either x is provided and y is not or y is provided and x is not.
        """
        # Get coordinates
        if x is None and y is None and z is None:
            x = self._x
            y = self._y
            z = self._z
        elif x is None:
            raise ValueError("Either all x, y, and z must be provided or none.")
        elif y is None:
            raise ValueError("Either all x, y, and z must be provided or none.")
        elif z is None:
            raise ValueError("Either all x, y, and z must be provided or none.")

        # Initialize center coordinates
        center_coords = []

        # Get face center coordinates by indices
        if face_type == "faces" and self._centers is not None:
            for i, j in self._centers:
                center_coords.append((x[i, j], y[i, j], z[i, j]))
        # Get face center coordinates as Center of Mass
        else:
            face_coords = self.get_face_coords(face_type=face_type, x=x, y=y, z=z)
            for coords in face_coords:
                # Calculate mean of x and y coordinates
                center_x = sum(c[0] for c in coords) / len(coords)
                center_y = sum(c[1] for c in coords) / len(coords)
                center_z = sum(c[2] for c in coords) / len(coords)
                center_coords.append((center_x, center_y, center_z))
        
        return center_coords


    def _get_face_center(
        self, 
        face: List[Tuple[int, int]], 
        x: Optional[np.ndarray] = None, 
        y: Optional[np.ndarray] = None,
        z: Optional[np.ndarray] = None
    ) -> Tuple[float, float, float]:
        """
        Get center coordinates of a face based on its indices.
        
        Parameters
        ----------
        face : List[Tuple[int, int]]
            The face to get center coordinates for.
        x : Optional[np.ndarray], default None
            X coordinates. If None, uses self._x.
        y : Optional[np.ndarray], default None
            Y coordinates. If None, uses self._y.
        z : Optional[np.ndarray], default None
            Z coordinates. If None, uses self._z.

        Returns 
        -------
        Tuple[float, float, float]
            The center coordinates of the face.
        """
        if x is None and y is None and z is None:
            x = self._x
            y = self._y
            z = self._z
        elif x is None:
            raise ValueError("Either all x, y, and z must be provided or none.")
        elif y is None:
            raise ValueError("Either all x, y, and z must be provided or none.")
        elif z is None:
            raise ValueError("Either all x, y, and z must be provided or none.")

        coords = [(x[j, i], y[j, i], z[j, i]) for i, j in face]
        center_x = sum(c[0] for c in coords) / len(coords)
        center_y = sum(c[1] for c in coords) / len(coords)
        center_z = sum(c[2] for c in coords) / len(coords)
        center = (center_x, center_y, center_z)

        return center


    def show_faces(
        self, 
        x: Optional[np.ndarray] = None, 
        y: Optional[np.ndarray] = None,
        z: Optional[np.ndarray] = None,
        face_type: Literal["faces", "triangles"] = "faces", 
        show_bounding_box: bool = True, 
        show_tiling_box: bool = False,
        show_centers: Optional[bool] = None, 
        show_edges: Optional[bool] = None, 
        face_color: Union[str, List[str]] = "#94C6E8", 
        padding_color: str = "#BBFF33",
        background_color: str = "white",
        alpha: Optional[float] = None,
        backend: Literal["matplotlib", "plotly"] = "plotly",
        view_position: Optional[Tuple[float, float, float]] = (10,180,0),
        view_vector: Optional[Tuple[float, float, float]] = (1,1,1),
        light_source: Optional[Tuple[float, float, float]] = (1,0,0),
        ambient_intensity: float = -0.1,
        diffuse_intensity: float = 0.2,
        specular_intensity: float = 0.1,
        shininess: float = 10.0,
        mode: Union[Literal["2D", "3D"], None] = None,
        zoom: Optional[float] = None,
        ax = None,
    ):
        """
        Show the grid faces.
        
        Note
        ----
        Parameters used only in "2D" mode:
        - "show_bounding_box", "show_tiling_box"
        Parameters used only in "3D" mode:
        - "view_position", "view_vector", "light_source", "ambient_intensity", 
        - "diffuse_intensity", "specular_intensity", "shininess", "backend"

        Parameters
        ----------
        x : Optional[np.ndarray], default None
            X coordinates. If None, uses self._x.
        y : Optional[np.ndarray], default None
            Y coordinates. If None, uses self._y.
        z : Optional[np.ndarray], default None
            Z coordinates. If None, uses self._z.
        face_type : Literal["faces", "triangles"], default "faces"
            Whether to show faces or triangles.
        show_bounding_box : bool, default True
            Whether to show the bounding box.
        show_tiling_box : bool, default False
            Whether to show the tiling box.
        show_centers : Optional[bool], default None
            Whether to show face centers as scatter points.
        show_edges : Optional[bool], default None
            Whether to show face edges.
        face_color : Union[str, List[str]], default "#94C6E8"
            Color for faces inside the bounding box. List of colors allowed for 3D.
        padding_color : str, default "#BBFF33"
            Color for faces outside the bounding box. List of colors allowed for 3D.
        background_color : str, default "white"
            Background color.
        alpha : Optional[float], default None
            Alpha value for the faces.
        backend : Literal["matplotlib", "plotly"], default "matplotlib"
            Backend to use for visualization. "plotly" provides interactive 3D visualization,
            while "matplotlib" provides static 2D visualization.
        view_position : Tuple[float, float, float], default (10,180,0)
            3D view elevation, azimuth, and roll angles.
        view_vector : Tuple[float, float, float], default (1,1,1)
            Specular effect direction of the view vector (x, y, z).
        light_source : Tuple[float, float, float], default (1,0,0)
            Position of light source (x, y, z).
        ambient_intensity : float, default -0.1
            Intensity of ambient light.
        diffuse_intensity : float, default 0.2
            Intensity of diffuse light.
        specular_intensity : float, default 0.1
            Intensity of specular light.
        shininess : float, default 10.0
            Shininess of the surface.
        mode : Union[Literal["2D", "3D"], None], default None
            If None, the mode is "3D" for spherical grids and "2D" for other grids.
        zoom : Optional[float], default None
            Zoom factor for the initial view. >1 zooms in, <1 zooms out.
        ax : Optional[matplotlib.axes.Axes], default None
            For 2D mode: Matplotlib axes to plot on. If None, creates a new figure and axes.

        Returns
        -------
        Union[matplotlib.axes.Axes, plotly.graph_objects.Figure]
            The visualization object.
        """
        # Automatically select mode
        if mode is None:
            if self._type == "spherical":
                mode = "3D"
            else:
                mode = "2D"

        # Automatically set show_centers, show_edges, alpha
        if mode == "2D":
            if show_centers is None:
                show_centers = True
            if show_edges is None:
                show_edges = True
            if alpha is None:
                alpha = 0.3
        elif mode == "3D":
            if show_centers is None:
                show_centers = False
            if show_edges is None:
                show_edges = False
            if alpha is None:
                alpha = 1.0

        # Show faces 3D
        if mode == "3D":
            if backend == "plotly" and PLOTLY_AVAILABLE:
                return self._show_3d_plotly(
                    x, y, z, show_centers, show_edges, face_color, padding_color, background_color, 
                    alpha, view_position, view_vector, light_source, ambient_intensity, diffuse_intensity, 
                    specular_intensity, shininess, zoom)
            else:
                return self._show_3d_matplotlib(
                    x, y, z, show_centers, show_edges, face_color, padding_color, background_color, 
                    alpha, view_position, view_vector, light_source, ambient_intensity, diffuse_intensity, 
                    specular_intensity, shininess, zoom)
        # Show faces 2D
        else:
            return self._show_2d(x, y, z, face_type, show_bounding_box, show_tiling_box, show_centers, 
                                 show_edges, face_color, padding_color, background_color, alpha, ax)
    

    def _show_2d(
        self,
        x: Optional[np.ndarray] = None, 
        y: Optional[np.ndarray] = None,
        z: Optional[np.ndarray] = None,
        face_type: Literal["faces", "triangles"] = "faces", 
        show_bounding_box: bool = True, 
        show_tiling_box: bool = False,
        show_centers: bool = True, 
        show_edges: bool = True, 
        face_color: Union[str, List[str]] = "#94C6E8", 
        padding_color: str = "#BBFF33",
        background_color: str = "white",
        alpha: float = 0.3,
        ax=None,
    ) -> matplotlib.axes.Axes:
        """Show 2D grid faces using matplotlib."""
        if ax is None:
            if not plt.get_fignums():
                _, ax = plt.subplots(figsize=(12,8))
            else:
                ax = plt.gca()
        plt.gcf().set_facecolor(background_color)
        
        # Get bounding box coordinates
        box_min, box_max = self._box
        x_min, y_min = box_min
        x_max, y_max = box_max
        
        # Get faces 2D coordinates
        if self._type == "spherical":
            x = self._phi if x is None else x
            y = self._theta if y is None else y
            z = self._r if z is None else z
        else:
            x = self._x if x is None else x
            y = self._y if y is None else y
            z = self._z if z is None else z
        face_coords, face_padding = self._get_face_coords_and_padding(face_type, x, y, z)
        coords2d = [coord[:2] for coord in face_coords]
            
        # Colors
        face_colors, dark_colors = self._set_face_colors(face_type, face_color, padding_color)

        # Show faces
        for coords, color, dark_color in zip(face_coords, face_colors, dark_colors):
            # Create polygon with appropriate color
            coords2d = [coord[:2] for coord in coords]
            edge_color = dark_color if show_edges else 'none'
            filled_polygon = Polygon(coords2d, facecolor=color, edgecolor=edge_color, alpha=alpha)
            ax.add_patch(filled_polygon)
            # Update x_min, y_min, x_max, y_max
            for coord in coords2d:
                x_min = min(x_min, coord[0])
                y_min = min(y_min, coord[1])
                x_max = max(x_max, coord[0])
                y_max = max(y_max, coord[1])

        # Plot face centers if requested
        if show_centers:
            centers = self.get_face_centers(face_type=face_type, x=x, y=y, z=z)
            center_x = [c[0] for c in centers]
            center_y = [c[1] for c in centers]
            ax.scatter(center_x, center_y, c=dark_colors, s=10, zorder=3)

        # Plot tiling box if requested
        if show_tiling_box and self._tiling_box is not None:
            tiling_min, tiling_max = self._tiling_box
            tiling_coords = [
                (tiling_min[0], tiling_min[1]),
                (tiling_max[0], tiling_min[1]),
                (tiling_max[0], tiling_max[1]),
                (tiling_min[0], tiling_max[1])
            ]
            tiling_polygon = Polygon(tiling_coords, facecolor='none', edgecolor='lightgray', linestyle='--', lw=2)
            ax.add_patch(tiling_polygon)

        # Plot bounding box if requested
        if show_bounding_box:
            box_coords = [
                (box_min[0], box_min[1]),
                (box_max[0], box_min[1]),
                (box_max[0], box_max[1]),
                (box_min[0], box_max[1])
            ]
            box_polygon = Polygon(box_coords, facecolor='none', edgecolor='gray', linestyle='--', lw=2)
            ax.add_patch(box_polygon)

        # Set equal aspect ratio
        ax.set_aspect('equal')

        # Set limits
        dx = x_max - x_min
        dy = y_max - y_min
        ax.set_xlim(x_min - 0.05 * dx, x_max + 0.05 * dx)
        ax.set_ylim(y_min - 0.05 * dy, y_max + 0.05 * dy)
        
        return ax


    def _show_3d_matplotlib(
        self, 
        x: Optional[np.ndarray] = None, 
        y: Optional[np.ndarray] = None,
        z: Optional[np.ndarray] = None,
        show_centers: bool = False, 
        show_edges: bool = False, 
        face_color: Union[str, List[str]] = "#94C6E8",
        padding_color: Union[str, List[str]] = "#BBFF33",
        background_color: str = "white",
        alpha: float = 1.0,
        view_position: Optional[Tuple[float, float, float]] = (10,180,0),
        view_vector: Optional[Tuple[float, float, float]] = (1,1,1),
        light_source: Optional[Tuple[float, float, float]] = (1,0,0),
        ambient_intensity: float = -0.1,
        diffuse_intensity: float = 0.2,
        specular_intensity: float = 0.1,
        shininess: float = 10.0,
        zoom: Optional[float] = None,
    ) -> matplotlib.axes.Axes:
        """
        Show 3D grid faces using matplotlib.

        Note
        ----
        Use %matplotlib widget magic command before calling this function
        to enable interactive 3D rotation in Jupyter Notebook.

        Parameters
        ----------
        x : Optional[np.ndarray], default None
            X coordinates. If None, uses self._x.
        y : Optional[np.ndarray], default None
            Y coordinates. If None, uses self._y.
        z : Optional[np.ndarray], default None
            Z coordinates. If None, uses self._z.
        show_centers : bool, default False
            Whether to show face centers as scatter points.
        show_edges : bool, default False
            Whether to show face edges.
        face_color : Union[str, List[str]], default "#94C6E8"
            Color for faces inside the bounding box. List of colors allowed for 3D.
        padding_color : Union[str, List[str]], default "#BBFF33"
            Color for faces outside the bounding box. List of colors allowed for 3D.
        background_color : str, default "white"
            Color for the plot background and axis planes
        alpha : float, default 1.0
            Alpha value for the faces.
        view_position : Tuple[float, float, float], default (10,180,0)
            3D view elevation, azimuth, and roll angles.
        view_vector : Tuple[float, float, float], default (1,1,1)
            Specular effect direction of the view vector (x, y, z).
        light_source : Tuple[float, float, float], default (1,0,0)
            Position of light source (x, y, z).
        ambient_intensity : float, default -0.1
            Intensity of ambient light
        diffuse_intensity : float, default 0.2
            Intensity of diffuse light
        specular_intensity : float, default 0.1
            Intensity of specular light
        shininess : float, default 10.0
            Surface shininess factor
        zoom : Optional[float], default None
            Zoom factor for the initial view. >1 zooms in, <1 zooms out.
        """
        # Set coordinates and zoom
        x = self._x if x is None else x
        y = self._y if y is None else y
        z = self._z if z is None else z
        zoom = 2.0 if zoom is None else zoom

        # Flatten coordinate arrays for vertex lookup
        vertices_x = x.flatten()
        vertices_y = y.flatten()
        vertices_z = z.flatten()
        
        # Convert face indices from 2D to 1D
        faces_1d = []
        n_cols = x.shape[1]
        for face in self._triangles:
            face_1d = [i * n_cols + j for (i, j) in face]
            faces_1d.append(face_1d)

        # Colors
        face_colors, dark_colors = self._set_face_colors(
            "triangles", face_color, padding_color, view_vector, light_source, ambient_intensity, 
            diffuse_intensity, specular_intensity, shininess, mode="3D"
        )

        # Create figure and axes
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Set figure and axes background color
        fig.patch.set_facecolor(background_color)
        ax.set_facecolor(background_color)
        
        # Create polygons for each face
        polygons = []
        for face in faces_1d:
            vertices = np.column_stack((
                vertices_x[face],
                vertices_y[face],
                vertices_z[face]
            ))
            polygons.append(vertices)
        
        # Create Poly3DCollection
        collection = Poly3DCollection(
            polygons,
            facecolor=face_colors,
            alpha=alpha,
            edgecolor=dark_colors if show_edges else 'none',
            linewidths=1 if show_edges else 0
        )
        ax.add_collection3d(collection)

        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        
        # Set axis limits to fit all data with a margin and zoom
        x_min, x_max = np.nanmin(vertices_x), np.nanmax(vertices_x)
        y_min, y_max = np.nanmin(vertices_y), np.nanmax(vertices_y)
        z_min, z_max = np.nanmin(vertices_z), np.nanmax(vertices_z)
        mid_x = (x_max + x_min) * 0.5
        mid_y = (y_max + y_min) * 0.5
        mid_z = (z_max + z_min) * 0.5
        max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / zoom  # <-- apply zoom
        margin = 0.05 * max_range
        ax.set_xlim(mid_x - max_range * 0.5 - margin, mid_x + max_range * 0.5 + margin)
        ax.set_ylim(mid_y - max_range * 0.5 - margin, mid_y + max_range * 0.5 + margin)
        ax.set_zlim(mid_z - max_range * 0.5 - margin, mid_z + max_range * 0.5 + margin)

        # Remove grid, ticks, labels
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_zlabel('')

        # Set pane colors to match background
        ax.xaxis.set_pane_color(to_rgb(background_color) + (1.0,))
        ax.yaxis.set_pane_color(to_rgb(background_color) + (1.0,))
        ax.zaxis.set_pane_color(to_rgb(background_color) + (1.0,))
        
        # Hide the axes lines
        ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        
        # Set the view position
        if view_position is not None:
            ax.view_init(elev=view_position[0], azim=view_position[1], roll=view_position[2])
        
        if show_centers:
            centers = self.get_face_centers(face_type="triangles", x=x, y=y, z=z)
            center_x = [c[0] for c in centers]
            center_y = [c[1] for c in centers]
            center_z = [c[2] for c in centers]
            # Use the same dark color as for edges
            _, dark_colors = self._set_face_colors("triangles", face_color, padding_color, view_vector, light_source, ambient_intensity, diffuse_intensity, specular_intensity, shininess, mode="3D")
            zorder = max([child.zorder for child in plt.gca().get_children()]) + 1
            ax.scatter(center_x, center_y, center_z, c=dark_colors, s=10, zorder=zorder, depthshade=False)
        
        return ax


    def _show_3d_plotly(
        self,
        x: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        z: Optional[np.ndarray] = None,
        show_centers: bool = False, 
        show_edges: bool = False, 
        face_color: Union[str, List[str]] = "#94C6E8",
        padding_color: Union[str, List[str]] = "#BBFF33",
        background_color: str = "white",
        alpha: float = 1.0,
        view_position: Optional[Tuple[float, float, float]] = (10,180,0),
        view_vector: Optional[Tuple[float, float, float]] = (1,1,1),
        light_source: Optional[Tuple[float, float, float]] = (1,0,0),
        ambient_intensity: float = -0.1,
        diffuse_intensity: float = 0.2,
        specular_intensity: float = 0.1,
        shininess: float = 10.0,
        zoom: Optional[float] = None,
    ) -> 'go.Figure':
        """
        Show spherical grid using Plotly's 3D capabilities.

        Parameters
        ----------
        x : Optional[np.ndarray], default None
            X coordinates. If None, uses self._x.
        y : Optional[np.ndarray], default None
            Y coordinates. If None, uses self._y.
        z : Optional[np.ndarray], default None
            Z coordinates. If None, uses self._z.
        show_centers : bool, default False
            Whether to show face centers as scatter points.
        show_edges : bool, default False
            Whether to show face edges.
        face_color : Union[str, List[str]], default "#94C6E8"
            Color for faces inside the bounding box. List of colors allowed for 3D.
        padding_color : Union[str, List[str]], default "#BBFF33"
            Color for faces outside the bounding box. List of colors allowed for 3D.
        background_color : str, default "white"
            Color for the plot background and axis planes
        alpha : float, default 1.0
            Alpha value for the faces.
        view_position : Tuple[float, float, float], default (10,180,0)
            3D view elevation, azimuth, and roll angles.
        view_vector : Tuple[float, float, float], default (1,1,1)
            Specular effect direction of the view vector (x, y, z).
        light_source : Tuple[float, float, float], default (1,0,0)
            Position of light source (x, y, z).
        ambient_intensity : float, default -0.1
            Intensity of ambient light
        diffuse_intensity : float, default 0.2
            Intensity of diffuse light
        specular_intensity : float, default 0.1
            Intensity of specular light
        shininess : float, default 10.0
            Surface shininess factor
        zoom : Optional[float], default None
            Zoom factor for the initial view. >1 zooms in, <1 zooms out.
        """
        # Set coordinates and zoom
        x = self._x if x is None else x
        y = self._y if y is None else y
        z = self._z if z is None else z
        zoom = 2.0 if zoom is None else zoom

        # Flatten coordinate arrays
        vertices_x = x.flatten()
        vertices_y = y.flatten()
        vertices_z = z.flatten()
        
        # Convert face indices from 2D to 1D
        faces_1d = []
        n_cols = x.shape[1]
        for face in self._triangles:
            face_1d = [i * n_cols + j for (i, j) in face]
            faces_1d.append(face_1d)
        
        # Colors
        face_colors, dark_colors = self._set_face_colors(
            "triangles", face_color, padding_color, view_vector, light_source, ambient_intensity, 
            diffuse_intensity, specular_intensity, shininess, mode="3D"
        )

        # Create figure
        init_notebook_mode(connected=True)
        fig = go.Figure()

        # Add surface mesh
        fig.add_trace(go.Mesh3d(
            x=vertices_x,
            y=vertices_y,
            z=vertices_z,
            i=[face[0] for face in faces_1d],
            j=[face[1] for face in faces_1d],
            k=[face[2] for face in faces_1d],
            facecolor=face_colors,
            opacity=alpha,
            showscale=False,
            flatshading=True,
            lighting=dict(ambient=1, diffuse=0, specular=0, roughness=1, fresnel=0)
        ))

        # Add edges if requested
        if show_edges:
            for i, face in enumerate(faces_1d):
                # Get edge color (darkened version of face color)
                edge_color = dark_colors[i]
                
                # Add three edges for each face
                for j in range(3):
                    fig.add_trace(go.Scatter3d(
                        x=[vertices_x[face[j]], vertices_x[face[(j+1)%3]]],
                        y=[vertices_y[face[j]], vertices_y[face[(j+1)%3]]],
                        z=[vertices_z[face[j]], vertices_z[face[(j+1)%3]]],
                        mode='lines',
                        line=dict(color=edge_color, width=2),
                        showlegend=False,
                        hoverinfo='skip'
                    ))

        # Add face centers if requested
        if show_centers:
            # Calculate face centers
            centers_x = []
            centers_y = []
            centers_z = []
            
            for face in faces_1d:
                # Calculate center of each face
                center_x = np.mean([vertices_x[i] for i in face])
                center_y = np.mean([vertices_y[i] for i in face])
                center_z = np.mean([vertices_z[i] for i in face])
                centers_x.append(center_x)
                centers_y.append(center_y)
                centers_z.append(center_z)
            
            # Add centers as scatter points
            fig.add_trace(go.Scatter3d(
                x=centers_x,
                y=centers_y,
                z=centers_z,
                mode='markers',
                marker=dict(
                    size=4,
                    color=dark_colors,
                    opacity=1.0,
                ),
                showlegend=False,
                hoverinfo='skip'
            ))

        # Compute data bounds for camera distance (using zoom parameter)
        x_min, x_max = np.nanmin(vertices_x), np.nanmax(vertices_x)
        y_min, y_max = np.nanmin(vertices_y), np.nanmax(vertices_y)
        z_min, z_max = np.nanmin(vertices_z), np.nanmax(vertices_z)
        max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
        camera_distance = 1.25 * max_range / zoom  # 1.25 is Plotly's default

        # Convert view position from matplotlib style (elev, azim, roll) to plotly camera position
        # Note: This is an approximation as coordinate systems differ
        elev, azim, roll = view_position
        # Convert angles to radians
        elev_rad = np.radians(elev)
        azim_rad = np.radians(azim)
        # Calculate camera position
        eye_x = camera_distance * np.cos(elev_rad) * np.sin(azim_rad)
        eye_y = camera_distance * np.cos(elev_rad) * np.cos(azim_rad)
        eye_z = camera_distance * np.sin(elev_rad)

        # Update layout for better visualization
        fig.update_layout(
            scene=dict(
                aspectmode='data',
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=eye_x, y=eye_y, z=eye_z)
                ),
                xaxis=dict(
                    showbackground=False,
                    showgrid=False,
                    showline=False,
                    showticklabels=False,
                    zeroline=False,
                    visible=False
                ),
                yaxis=dict(
                    showbackground=False,
                    showgrid=False,
                    showline=False,
                    showticklabels=False,
                    zeroline=False,
                    visible=False
                ),
                zaxis=dict(
                    showbackground=False,
                    showgrid=False,
                    showline=False,
                    showticklabels=False,
                    zeroline=False,
                    visible=False
                ),
                bgcolor=background_color
            ),
            paper_bgcolor=background_color,
            plot_bgcolor=background_color,
            margin=dict(l=0, r=0, b=0, t=0),
            showlegend=False
        )

        fig.show()
        return fig


    def _set_face_colors(
        self,
        face_type: Literal["faces", "triangles"] = "faces", 
        face_color: Union[str, List[str]] = "#94C6E8", 
        padding_color: str = "#BBFF33",
        view_vector: Optional[Tuple[float, float, float]] = (1,1,1),
        light_source: Optional[Tuple[float, float, float]] = (1,0,0),
        ambient_intensity: float = -0.1,
        diffuse_intensity: float = 0.2,
        specular_intensity: float = 0.1,
        shininess: float = 10.0,
        mode: Literal["2D", "3D"] = "2D",
    ):
        """
        Set face colors.

        Parameters
        ----------
        face_type : Literal["faces", "triangles"], default "faces"
            Whether to set colors for faces or triangles.
        face_color : Union[str, List[str]], default "#94C6E8"
            Color for faces inside the bounding box. List of colors allowed for 3D.
        padding_color : str, default "#BBFF33"
            Color for faces outside the bounding box.
        view_vector : Tuple[float, float, float], default (1,1,1)
            Specular effect direction of the view vector (x, y, z).
        light_source : Tuple[float, float, float], default (1,0,0)
            Position of light source (x, y, z).
        ambient_intensity : float, default -0.1
            Intensity of ambient light.
        diffuse_intensity : float, default 0.2
            Intensity of diffuse light.
        specular_intensity : float, default 0.1
            Intensity of specular light.
        shininess : float, default 10.0
            Shininess of the surface.
        mode: Literal["2D", "3D"] = "2D",
            Mode to set colors for.

        Returns
        -------
        face_colors : List[str]
            List of face colors.
        dark_colors : List[str]
            List of darker face colors for edges or centers.
        """
        # Padding flag
        _, is_padding_face = self._get_face_coords_and_padding(face_type)

        # Initialize face colors
        face_colors = []
        if isinstance(face_color, list):
            if len(face_color) != len(is_padding_face):
                raise ValueError("Number of face_color must be str OR match the number of faces.")
            face_colors = [color for color in face_color]
        else:
            for is_padding in is_padding_face:
                face_colors.append(face_color if not is_padding else padding_color)

        # Lighting
        if mode == "3D":
            face_normals = _calc_normals(self._x, self._y, self._z, self._triangles)
            face_colors = _adjust_lighting(
                face_normals, light_source, view_vector, face_colors,
                ambient_intensity, diffuse_intensity, specular_intensity, shininess
            )

        # Dark colors
        dark_colors = [adjust_color(color, -0.3) for color in face_colors]

        return face_colors, dark_colors


    def show_triangles(
        self, 
        x: Optional[np.ndarray] = None, 
        y: Optional[np.ndarray] = None,
        z: Optional[np.ndarray] = None,
        show_bounding_box: bool = True,
        show_tiling_box: bool = False,
        show_centers: bool = True,
        face_color: Union[str, List[str]] = "#94C6E8",
        padding_color: Union[str, List[str]] = "#BBFF33",
        backend: Literal["matplotlib", "plotly"] = "matplotlib",
        view_position: Optional[Tuple[float, float, float]] = (30, 45, 0),
        view_vector: Tuple[float, float, float] = (0,0,1),
        light_source: Tuple[float, float, float] = (1,1,1),
        ambient_intensity: float = -0.1,
        diffuse_intensity: float = 0.2,
        specular_intensity: float = 0.1,
        shininess: float = 10.0,
        mode: Union[Literal["2D", "3D"], None] = None,
        ax=None,
    ):
        """
        Show triangular faces of the grid.
        This is a convenience wrapper around show_faces() with face_type="triangles".
        
        Parameters
        ----------
        See show_faces() for parameter descriptions.
        """
        return self.show_faces(
            x=x,
            y=y,
            z=z,
            face_type="triangles",
            show_bounding_box=show_bounding_box,
            show_tiling_box=show_tiling_box,
            show_centers=show_centers,
            face_color=face_color,
            padding_color=padding_color,
            backend=backend,
            view_position=view_position,
            view_vector=view_vector,
            light_source=light_source,
            ambient_intensity=ambient_intensity,
            diffuse_intensity=diffuse_intensity,
            specular_intensity=specular_intensity,
            shininess=shininess,
            mode=mode,
            ax=ax,
        )


    def get_tiling(self) -> Tuple[float, float]:
        """
        Get the tiling dx and dy.
        """
        if self._tiling_box is None:
            return None
        dx = self._tiling_box[1][0] - self._tiling_box[0][0]
        dy = self._tiling_box[1][1] - self._tiling_box[0][1]
        return dx, dy


    def get_info(self) -> None:
        """
        Print information about the grid attributes.
        
        If no grid has been initialized, prints a message indicating that.
        Otherwise, prints all relevant grid attributes in a readable format.
        """
        if self._noise is None:
            print("NO GRID HAS BEEN INITIALIZED YET")
            return
            
        print("\n2D Grid Information:")
        print("-" * 25)
        
        # Basic grid properties
        print(f'Grid Type: "{self._type.title()}"')
        if self._type in ["triangular", "hexagonal"]:
            print(f"Base: {self._base.title()}")
            print(f"Tilt: {self._tilt.title()}")
        print(f"Padding: {self._padding}")
        print(f"Shape: {self._shape[:2]}")
        print(f"Density: {self._dens[:2]}")
        print(f"Size: {self._size[:2]}")
        print(f"Stride: {self._stride}")
        
        
        # Bounding box
        if self._box is not None:
            print("\nBounding Box:")
            print(f"Size: {float(np.round(self._box[1][0] - self._box[0][0], 3))}, {float(np.round(self._box[1][1] - self._box[0][1], 3))}")
            print(f"Min: {float(np.round(self._box[0][0], 3))}, {float(np.round(self._box[0][1], 3))}")
            print(f"Max: {float(np.round(self._box[1][0], 3))}, {float(np.round(self._box[1][1], 3))}")
            print(f"Center: {self._center[:2]}")
            if self._tiling_box is not None:
                tiling_x = self._tiling_box[1][0] - self._tiling_box[0][0]
                tiling_y = self._tiling_box[1][1] - self._tiling_box[0][1]
                print(f"Tiling: X: {tiling_x}, Y: {tiling_y}")
        
        # Grid dimensions
        if self._x is not None:
            print("\nGrid Dimensions:")
            print(f"X shape: {self._x.shape}")
            print(f"Y shape: {self._y.shape}")
            if self._z is not None:
                print(f"Z shape: {self._z.shape}")
        
        # Noise parameters
        print(f"\nNoise Parameters:")
        print(f"Noise size: {self._noise.shape}")
        print(f"Input shape: {self._shape}")
        print(f"Input dens: {self._dens}")
        print(f"Octaves: {self._octaves}")
        print(f"Seed: {self._seed}")
        print(f"Warp: {self._warp}")

        # Face information
        if self._faces is not None:
            print(f"\nNumber of Faces: {len(self._faces)}")
            print(f"Number of Triangles: {len(self._triangles) if self._triangles is not None else 0}")
        
        print("-" * 25)

        return


    def get_noise(self) -> np.ndarray:
        """
        Get the Perlin noise grid values.
        
        Returns
        -------
        np.ndarray
            Array of grid values.
            
        Raises
        ------
        ValueError
            If no grid has been initialized.
        """
        if self._noise is None:
            raise ValueError("No grid has been initialized.")
        return np.copy(self._noise)


    def get_coords(self, mode: Literal["cartesian", "spherical"] = "cartesian") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get all 2d coordinate arrays (x, y, z) or (r, theta, phi) of the grid.
        
        Parameters
        ----------
        mode : Literal["cartesian", "spherical"], optional
            The coordinate system to return. Default is "cartesian".
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Tuple containing arrays of x, y, and z coordinates.
            
        Raises
        ------
        ValueError
            If no grid has been initialized.
        """
        if self._x is None or self._y is None or self._z is None:
            raise ValueError("No grid has been initialized.")
        
        if mode == "cartesian":
            return np.copy(self._x), np.copy(self._y), np.copy(self._z)
        elif mode == "spherical":
            if self._r is None or self._theta is None or self._phi is None:
                r, theta, phi = cartesian_to_spherical(self._x, self._y, self._z)
                if any([c != 0 for c in self._center]):
                    warnings.warn("Note that the center is not at the origin, so spherical coordinates are not centered.")
            else:
                # If the spherical coordinates are already computed, they are centered
                r, theta, phi = np.copy(self._r), np.copy(self._theta), np.copy(self._phi)
            return r, theta, phi









