#!/usr/bin/env python
# -*- coding: utf8 -*-

import numpy as np
import itertools
import warnings
from typing import Union, Tuple, List, Optional, Literal
from math import sqrt, acos, pi
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
    
    def __init__(self, shape: Optional[Tuple[int, ...]] = None,
                 size: Optional[Tuple[int, ...]] = None,
                 padding: Optional[Tuple[int, ...]] = (0, 0),
                 grid_type: Optional[Literal["rectangular", "triangular", "hexagonal", "spherical"]] = "rectangular",
                 orientation: Optional[Literal["horizontal", "vertical"]] = "horizontal"):
        """
        Initialize a SurfaceGrid instance.
        
        Parameters
        ----------
        shape : Optional[Tuple[int, ...]]
            Shape of the grid (e.g., (16, 9) for a 16x9 grid).
        size : Optional[Tuple[int, ...]]
            Size covered by the grid (e.g., (1280, 720) for a 1280x720 grid).
        padding : Optional[Tuple[int, ...]]
            Padding, e.g., (1, 1) for 1 tile padding in each direction.
        grid_type : Optional[Literal["rectangular", "triangular", "hexagonal", "spherical"]]
            Type of the grid to create.
        orientation : Optional[Literal["horizontal", "vertical"]]
            Orientation of the triangle or hexagons base
            
        Raises
        ------
        ValueError: If both grid and other parameters are provided, or if grid is not 2D.
        """
        self._x = None
        self._y = None
        self._z = None
        self._grid = None
        self._type = None
        self._size = None
        self._dens = None
        self._shape = None
        self._padding = None
        self._base = None
        self._tilt = None
        self._faces = None
        self._triangles = None

        if shape is not None:
            if grid_type == "rectangular":
                self.make_rectangular_grid(*shape, size=size, padding=padding)
            elif grid_type == "triangular":
                self.make_triangular_grid(*shape, size=size, padding=padding, orientation=orientation)
            elif grid_type == "hexagonal":
                self.make_hexagonal_grid(*shape, size=size, padding=padding, orientation=orientation)
            elif grid_type == "spherical":
                self.make_spherical_grid(*shape, size=size, padding=padding)
        

    def make_rectangular_grid(
        self,
        *shape: Union[int, Tuple[int, ...]],
        dens: Union[int, Tuple[int, ...]] = 1,
        center: Optional[Tuple[int, int]] = None,
        stride: float = 1.0,
        padding: bool = False,
    ) -> None:
        """
        Generate a 2D rectangular grid.
        
        Parameters
        ----------
        *shape : Union[int, Tuple[int, ...]]
            Shape of the major grid nodes. Can be provided as separate integers or a tuple.
        dens : int, default 1
            Number of grid nodes from one major node to the next along each axis. Positive integer.
        center : Optional[Tuple[int, int]], default None
            Center of the grid. If None, the center of the grid is the center of the shape.
        stride : float, default 1.0
            Stride of the grid.
        padding : bool, default False
            If True, pad the grid with one row and one column in each direction.
 
        Raises
        ------
        Warning: If grid already exists
        ValueError: If shape and dens have different numbers of elements or are not 2D or 3D
        ValueError: If maximum grid nodes in 2D plane is exceeded
        """
        # Convert shape and dens to tuples
        shape, dens = _shape_and_dens_to_tuples(shape, dens=dens)
        if len(shape) != len(dens) or len(shape) not in (2, 3):
            raise ValueError("Shape and dens must have the same number of elements and must be 2D or 3D.")
        
        if self._grid is not None:
            warnings.warn("Grid already exists. Skipping grid generation.")
            return
        
        # Assign grid type
        self._type = "rectangular"

        # Calculate grid center and bounding box
        size = tuple(s * d for s, d in zip(shape, dens))
        x0 = 0.5 * size[0] * stride
        y0 = 0.5 * size[1] * stride

        # Assign center if not provided
        if center is not None:
            self._center = center
        else:
            self._center = (x0, y0)

        # Assign bounding box
        self._box = ((self._center[0] - x0, self._center[1] - y0), (self._center[0] + x0, self._center[1] + y0))
            
        # Calculate the size of the grid
        self._shape = shape
        self._dens = dens
        self._size = tuple((s + 2 * int(padding)) * d for s, d in zip(shape, dens))
        if np.prod(self._size[0:2]) > 1000:
            raise ValueError(f'Maximum grid nodes in 2D plane is 1000. Provided shape, dens, and padding produce {np.prod(self._size[0:2])} grid nodes in 2D plane.')

        # Create the grid
        x, y = [np.linspace(0, s, s + 1) for s in self._size[0:2]]
        x, y = np.meshgrid(x, y)

        # Pad the grid
        self._padding = padding
        if padding:
            x -= self._dens[0]
            y -= self._dens[1]

        # Assign grid coordinates
        self._x = x * stride + self._center[0]
        self._y = y * stride + self._center[1]
        self._z = np.zeros_like(self._x)

        # Assign indices of the grid faces
        self._faces = []
        for i in range(x.shape[0] - 1):
            for j in range(x.shape[1] - 1):
                self._faces.append([(i, j), (i+1, j), (i+1, j+1), (i, j+1)])

        # Assign indices of the grid triangles
        self._triangles = []
        for i in range(x.shape[0] - 1):
            for j in range(x.shape[1] - 1):
                self._triangles.append([(i, j), (i+1, j), (i+1, j+1)])
                self._triangles.append([(i, j), (i+1, j+1), (i, j+1)])

        return


    def make_triangular_grid(
        self,
        *shape: Union[int, Tuple[int, ...]],
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
        *shape : Union[int, Tuple[int, ...]]
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
        if len(shape) != len(dens) or len(shape) not in (2, 3):
            raise ValueError("Shape and dens must have the same number of elements and must be 2D or 3D.")
        
        if self._grid is not None:
            warnings.warn("Grid already exists. Skipping grid generation.")
            return
        
        # Assign grid type
        self._type = "triangular"
        
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
        if center is not None:
            self._center = center
        else:
            self._center = (x0, y0)

        # Assign bounding box
        self._box = ((self._center[0] - x0, self._center[1] - y0), (self._center[0] + x0, self._center[1] + y0))

        # Calculate the size of the grid
        self._shape = shape
        self._dens = dens
        self._size = tuple((s + 2 * int(padding)) * d for s, d in zip(shape, dens))
        if np.prod(self._size[0:2]) > 1000:
            raise ValueError(f'Maximum grid nodes in 2D plane is 1000. Provided shape, dens, and padding produce {np.prod(self._size[0:2])} grid nodes in 2D plane.')

        # Create the grid
        x, y = [np.linspace(0, s, s + 1) for s in self._size[0:2]]
        x, y = np.meshgrid(x, y)

        # Pad the grid
        self._tilt = tilt
        self._base = base
        self._padding = padding
        if self._padding:
            x -= self._dens[0]
            y -= self._dens[1]
            if self._base == 'horizontal' and self._dens[1] % 2 == 1:
                self._tilt = _reverse_tilt(self._tilt)
            if self._base == 'vertical' and self._dens[0] % 2 == 1:
                self._tilt = _reverse_tilt(self._tilt)

        if self._base == 'horizontal':
            y *= np.sqrt(3) / 2
            x = _tilt_axis(x, self._tilt)
            y[np.isnan(x)] = np.nan
        elif self._base == 'vertical':
            x *= np.sqrt(3) / 2
            y = _tilt_axis(y.T, self._tilt).T
            x[np.isnan(y)] = np.nan

        # Assign grid coordinates
        self._x = x * stride + self._center[0]
        self._y = y * stride + self._center[1]
        self._z = np.zeros_like(self._x)

        # Assign indices of the grid faces
        self._faces = []
        for i in range(self._x.shape[0] - 1):
            for j in range(self._x.shape[1] - 1):
                level = i + int(self._tilt in ['backward', 'outward'])
                if self._base == 'vertical':
                    level = j + int(self._tilt in ['backward', 'outward'])
                if level % 2 == 0:
                    if np.isfinite(self._x[i+1, j+1]):
                        self._faces.append([(i, j), (i+1, j), (i+1, j+1)])
                        if np.isfinite(self._x[i, j+1]):
                            self._faces.append([(i, j), (i+1, j+1), (i, j+1)])
                else:
                    if np.isfinite(self._x[i, j+1]):
                        self._faces.append([(i, j), (i+1, j), (i, j+1)])
                        if np.isfinite(self._x[i+1, j+1]):
                            self._faces.append([(i+1, j), (i+1, j+1), (i, j+1)])

        # Assign indices of the grid triangles
        self._triangles = self._faces
            
        return


    def get_face_coordinates(
        self, 
        faces: Literal["faces", "triangles"] = "faces", 
        x: Optional[np.ndarray] = None, 
        y: Optional[np.ndarray] = None
    ) -> List[List[Tuple[float, float]]]:
        """Get coordinates of faces based on their indices.
        
        Parameters
        ----------
        faces : Literal["faces", "triangles"], default "faces"
            Whether to return faces or triangles.
        x : Optional[np.ndarray], default None
            X coordinates. If None, uses self._x.
        y : Optional[np.ndarray], default None
            Y coordinates. If None, uses self._y.
            
        Returns
        -------
        List[List[Tuple[float, float]]]
            List of face coordinates, where each face is a list of (x,y) coordinate tuples.
            
        Raises
        ------
        ValueError: If either x is provided and y is not or y is provided and x is not.
        """
        # Use stored faces if none provided
        if faces == "faces":
            faces = self._faces
        elif faces == "triangles":
            faces = self._triangles
        
        # Use stored coordinates if none provided
        if x is None and y is None:
            x = self._x
            y = self._y
        elif x is None:
            raise ValueError("Either both x and y must be provided or none.")
        elif y is None:
            raise ValueError("Either both x and y must be provided or none.")
        
        # Get coordinates for each face
        face_coords = []
        for face in faces:
            coords = [(x[i, j], y[i, j]) for i, j in face]
            face_coords.append(coords)
        
        return face_coords


    def show_faces(self, ax=None, show_box=True, main_color='#94C6E8', padding_color='#EEFFCC'):
        """
        Plot the grid faces with different colors for main and padding faces.
        
        Parameters
        ----------
        ax : Optional[matplotlib.axes.Axes], default None
            Matplotlib axes to plot on. If None, creates a new figure and axes.
        show_box : bool, default True
            Whether to show the bounding box.
        main_color : str, default '#94C6E8'
            Color for faces inside the bounding box.
        padding_color : str, default '#EEFFCC'
            Color for faces outside the bounding box.
            
        Returns
        -------
        matplotlib.axes.Axes
            The axes object with the plot.
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 10))
        
        # Get bounding box coordinates
        box_min, box_max = self._box
        
        # Get faces coordinates
        face_coords = self.get_face_coordinates(faces="faces", x=self._x, y=self._y)
        
        # Plot faces
        for face in face_coords:
            # Get face coordinates
            if self._type == "rectangular":
                coords = [(self._x[i, j], self._y[i, j]) for i, j in face]
            else:  # triangular
                coords = [(self._x[i, j], self._y[i, j]) for i, j in face]
            
            # Check if face is inside bounding box
            is_inside = all(box_min[0] <= x <= box_max[0] and box_min[1] <= y <= box_max[1] 
                           for x, y in coords)
            
            # Create polygon with appropriate color
            color = main_color if is_inside else bounding_color
            polygon = Polygon(coords, facecolor=color, edgecolor='black', alpha=0.7)
            ax.add_patch(polygon)
        
        # Plot bounding box if requested
        if show_box:
            box_coords = [
                (box_min[0], box_min[1]),
                (box_max[0], box_min[1]),
                (box_max[0], box_max[1]),
                (box_min[0], box_max[1])
            ]
            box_polygon = Polygon(box_coords, facecolor='none', edgecolor='red', linestyle='--')
            ax.add_patch(box_polygon)
        
        # Set equal aspect ratio and limits
        ax.set_aspect('equal')
        ax.set_xlim(box_min[0] - 1, box_max[0] + 1)
        ax.set_ylim(box_min[1] - 1, box_max[1] + 1)
        
        return ax




