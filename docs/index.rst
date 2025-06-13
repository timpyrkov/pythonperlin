.. pythonperlin documentation master file, created by
   sphinx-quickstart on Fri Mar 18 23:10:45 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:

   pixel_maps/index
   surface/index

PythonPerlin
============

Perlin noise generator | Worley noise generator | Python | N-dimensional | Seamless tiling | Domain warping | Octaves | 2D and 3D surface grids
-----------------------------------------------------------------------------------------------------------------------------------------------

Perlin noise is a type of gradient noise suggested by professor Ken Perlin in 1983 at the Department of Computer Science at New York University (NYU).

Ken Perlin started developing the Perlin noise algorithm during his work on Disney's 1982 science fiction film "Tron". The movie starred Bruce Boxleitner and Peter Jurasik, the two, who later appeared among mian characters in the 1993-1998 TV-series "Babylon-5".

In 1997, Ken Perlin received an Academy Award for Technical Achievement from the Academy of Motion Picture Arts and Sciences, recognizing his development of this widely used technique for procedural generation of natural-looking textures. 

Worley noise is a type of cellular noise suggested by Steven Worley in 1996. In 2002, Steven Worley and Ken Perlin, along with other notable authors contributed to the book "Texturing and Modeling: A Procedural Approach".

This documentation shows and explains procedural generation of:
- Clouds, fire, water, wood, marble with Perlin noise
- Bubbles and cobblestones with Worley noise
- Terrain and planets with triangulated surface grids




.. image:: images/perlin_noise.png
   :width: 50%
   :align: center
   :alt: Perlin noise example




.. image:: images/perlin_noise.png
   :width: 50%
   :align: center
   :alt: Perlin noise example




Installation
------------

::

   pip install pythonperlin



Quick start
-----------

::

   import pylab as plt
   import pythonperlin as pp

   dens = 32
   shape = (8,8)
   x = pp.perlin(shape, dens=dens)
   plt.imshow(x, cmap=plt.get_cmap('viridis'))


Pixel Maps
----------

The pixel maps module provides functions for generating Perlin noise in 2D and 3D, which can be used to create various procedural textures and patterns.

.. toctree::
   :maxdepth: 2
   :caption: Pixel Maps:
   :hidden:

   pixel_maps/functions
   pixel_maps/examples
   pixel_maps/more_examples

Surface Faces
-------------

The surface module provides tools for generating and visualizing 3D surfaces using Perlin noise, supporting various grid types and visualization methods.

.. toctree::
   :maxdepth: 2
   :caption: Surface Faces:
   :hidden:

   surface/functions
   surface/pipeline
   surface/examples

