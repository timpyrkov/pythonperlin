[![Python Versions](https://img.shields.io/pypi/pyversions/pythonperlin?style=plastic)](https://pypi.org/project/pythonperlin/)
[![PyPI](https://img.shields.io/pypi/v/pythonperlin?style=plastic)](https://pypi.org/project/pythonperlin/)
[![License](https://img.shields.io/pypi/l/pythonperlin?style=plastic)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/pythonperlin/badge/?version=latest)](https://pythonperlin.readthedocs.io/en/latest/?badge=latest)

# PythonPerlin
## Perlin noise in python - seamlessly tile in any dimensions
#

# Installation
```
pip install pythonperlin
```

# Generate Perlin noise with octaves
```
import pylab as plt
from pythonperlin import perlin

# Set grid shape for randomly seeded gradients
shape = (8,8)

# Set density - output shape will be shape * dens = (256,256)
dens = 32

# Generate noise
x = perlin(shape, dens=dens, seed=0)

plt.figure(figsize=(6,6))
plt.imshow(x, cmap=plt.get_cmap('viridis'))
plt.axis('off')
plt.show()

# Generate noise array with 2 additional octaves
x = perlin(shape, dens=dens, seed=0, octaves=2)

plt.figure(figsize=(6,6))
plt.imshow(x, cmap=plt.get_cmap('viridis'))
plt.axis('off')
plt.show()
```
![](img/img_no_octaves.png) ![](img/img_with_octaves.png)

# Generate water caustics
```
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
```
![](img/img_caustics.png)

# Generate flower petals
```
dens = 32
shape = (8,8)
x = perlin(shape, dens=dens)

n = 8
delta = dens
plt.figure(figsize=(6,6))
for i in range(n):
    r = x[delta * i] + 1
    r = np.concatenate([r, (r[0],)])
    phi = 2 * np.pi * np.linspace(0, 1, len(r))
    scale = 1 - i / n
    z = scale * r * np.exp(1j * phi)
    ax = plt.gca()
    zorder = max([ch.zorder for ch in ax.get_children()])
    plt.fill(z.real, z.imag, c='dodgerblue', zorder=zorder+1)
    plt.plot(z.real, z.imag, c='blue', lw=2, zorder=zorder+2)
plt.axis('off')
plt.show()
```
![](img/img_flower.png)

# Generate vector field
```
dens = 6
shape = (3,3)
x = perlin(shape, dens=dens)
z = np.exp(2j * np.pi * x)

shape = z.shape
plt.figure(figsize=(6,6))
for i in range(shape[0]):
    for j in range(shape[1]):
        di = 0.5 * z[i,j].real
        dj = 0.5 * z[i,j].imag
        plt.arrow(i, j, di, dj, color='dodgerblue', width=0.1)
plt.axis('off')
plt.show()
```
![](img/img_vectors.png)
# Documentation

[https://pythonperlin.readthedocs.io](https://pythonperlin.readthedocs.io)