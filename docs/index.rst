.. pythonperlin documentation master file, created by
   sphinx-quickstart on Fri Mar 18 23:10:45 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PythonPerlin
============

Python implementation of Perlin noise - to seamlessly tile in any dimensions


Installation
------------

::

   pip install pythonperlin



Quick start
-----------

::

   import pylab as plt
   from pythonperlin import perlin

   dens = 32
   shape = (8,8)
   x = perlin(shape, dens=dens)
   plt.imshow(x, cmap=plt.get_cmap('viridis'))


.. toctree::
   :maxdepth: 2
   :caption: Examples

   notebook/example

