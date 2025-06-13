Usage Pipeline
=============

This section describes the typical workflow for using the SurfaceGrid class.

Initialization
-------------

.. image:: ../img/scheme1.jpg
   :alt: Surface class initialization methods
   :align: center

The SurfaceGrid class can be initialized with various parameters to create different types of grids:

1. Rectangular Grid
2. Triangular Grid
3. Hexagonal Grid
4. Cellular Grid
5. Spherical Grid

Getter Methods
-------------

.. image:: ../img/scheme2.jpg
   :alt: Getter methods for class attributes
   :align: center

The SurfaceGrid class provides several methods to access and calculate surface properties:

1. get_face_coords() - Get coordinates of faces
2. get_face_idxs() - Get face indices
3. get_face_normals() - Calculate face normals
4. get_face_centers() - Calculate face centers
5. get_info() - Get grid information
6. get_noise() - Get noise values
7. get_coords() - Get coordinate arrays

Data Structure
-------------

.. image:: ../img/scheme3.jpg
   :alt: Scheme of 2d coordinates arrays and surface faces
   :align: center

The surface data is organized into:

1. 2D coordinate arrays (x, y)
2. 3D noise array
3. Surface faces (triangles or polygons)

Visualization
------------

.. image:: ../img/scheme4.jpg
   :alt: Pipeline to visualize surface faces
   :align: center

The SurfaceGrid class provides several visualization methods:

1. 2D Visualization
   - show_faces() with mode="2D"
   - show_triangles() with mode="2D"

2. 3D Visualization
   - show_faces() with mode="3D"
   - show_triangles() with mode="3D"

3. Custom Visualization
   - Use get_face_coords() and get_face_normals() for custom visualization
   - Support for matplotlib and plotly backends 