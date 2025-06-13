Examples
========

This section contains examples of using Perlin noise for generating various pixel maps.

Basic Usage
----------

Generate a simple Perlin noise pattern:

.. code-block:: python

   import pylab as plt
   from pythonperlin import perlin

   # Set grid shape for randomly seeded gradients
   shape = (4,4)

   # Set density - output shape will be dens * shape = (128,128)
   dens = 32

   # Generate noise
   x = perlin(shape, dens=dens, seed=0)

   # Test that noise tiles seamlessly
   x = np.concatenate([x] * 2, axis=1)

   plt.figure(figsize=(12,6))
   plt.imshow(x, cmap=plt.get_cmap('Accent_r'))
   plt.axis('off')
   plt.show()

.. image:: ../../media/img_tile.png
   :alt: Tiled Perlin noise
   :align: center

Domain Warping
-------------

Add noise to grid coordinates and generate noise again:

.. code-block:: python

   dens = 32
   shape = (4,4)
   x = perlin(shape, dens=dens, seed=0, warp=2)

   plt.figure(figsize=(6,6))
   plt.imshow(x, cmap=plt.get_cmap('Accent_r'))
   plt.axis('off')
   plt.show()

.. image:: ../../media/img_warp.png
   :alt: Domain warped Perlin noise
   :align: center

Octaves
-------

Generate noise with multiple octaves for more detail:

.. code-block:: python

   import pylab as plt
   from pythonperlin import perlin

   # Set grid shape for randomly seeded gradients
   shape = (8,8)

   # Set density - output shape will be shape * dens = (256,256)
   dens = 32

   # Generate noise without octaves
   x = perlin(shape, dens=dens, seed=0)

   plt.figure(figsize=(6,6))
   plt.imshow(x, cmap=plt.get_cmap('Accent_r'))
   plt.axis('off')
   plt.show()

   # Generate noise array with 4 additional octaves
   x = perlin(shape, dens=dens, seed=0, octaves=4)

   plt.figure(figsize=(6,6))
   plt.imshow(x, cmap=plt.get_cmap('Accent_r'))
   plt.axis('off')
   plt.show()

.. image:: ../../media/img_no_octaves.png
   :alt: Perlin noise without octaves
   :align: center

.. image:: ../../media/img_with_octaves.png
   :alt: Perlin noise with octaves
   :align: center

Water Caustics
-------------

Take absolute value of Perlin noise and apply log-scaled color gradient:

.. code-block:: python

   import numpy as np
   from matplotlib.colors import LinearSegmentedColormap

   dens = 32
   shape = (8,8)
   x = perlin(shape, dens=dens)

   # Take absolute values of Perlin noise
   x = np.abs(x)

   # Log-scale colormap
   logscale = np.logspace(0,-3,5)
   colors = plt.cm.get_cmap('GnBu_r')(logscale)
   cmap = LinearSegmentedColormap.from_list('caustics', colors)

   plt.figure(figsize=(6,6))
   plt.imshow(x, cmap=cmap)
   plt.axis('off')
   plt.show()

.. image:: ../../media/img_caustics.png
   :alt: Water caustics effect
   :align: center

Flower Petals
------------

Take 1D Perlin noise as the varying radius along a circle:

.. code-block:: python

   dens = 32
   shape = (8,8)
   x = perlin(shape, dens=dens)

   n = 8
   delta = dens
   color = plt.get_cmap('tab20').colors[::-1]
   plt.figure(figsize=(6,6))
   for i in range(n):
       r = x[delta * i] + 1
       r = np.concatenate([r, (r[0],)])
       phi = 2 * np.pi * np.linspace(0, 1, len(r))
       scale = 1 - i / (n + 2)
       z = scale * r * np.exp(1j * phi)
       ax = plt.gca()
       zorder = max([ch.zorder for ch in ax.get_children()])
       plt.fill(z.real, z.imag, c=color[2*i], zorder=zorder+1)
       plt.plot(z.real, z.imag, c=color[2*i+1], lw=2, zorder=zorder+2)
   plt.axis('off')
   plt.show()

.. image:: ../../media/img_flower.png
   :alt: Flower petals effect
   :align: center

Vector Field
-----------

Take Perlin noise as the vector angle at each point of a grid:

.. code-block:: python

   dens = 6
   shape = (3,3)
   x = perlin(shape, dens=dens)
   z = np.exp(2j * np.pi * x)

   shape = z.shape
   colors = plt.get_cmap('Accent').colors
   plt.figure(figsize=(6,6))
   for i in range(shape[0]):
       for j in range(shape[1]):
           di = 0.5 * z[i,j].real
           dj = 0.5 * z[i,j].imag
           color = colors[(di > 0) + 2 * (dj > 0)]
           plt.arrow(i, j, di, dj, color=color, width=0.1)
   plt.axis('off')
   plt.show()

.. image:: ../../media/img_vectors.png
   :alt: Vector field
   :align: center

Audio Generation
--------------

Perlin noise can be used to generate audio:

.. code-block:: python

   import sounddevice as sd

   dens = 32
   shape = (1024,)
   x = perlin(shape, dens=dens)

   sd.play(x, 22050)

Alternatively, save and play as a WAV file:

.. code-block:: python

   import IPython
   import soundfile as sf

   sf.write('perlin.wav', x, 22050)
   IPython.display.Audio('perlin.wav') 