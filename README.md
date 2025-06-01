[![Python Versions](https://img.shields.io/pypi/pyversions/pythonperlin?style=plastic)](https://pypi.org/project/pythonperlin/)
[![PyPI](https://img.shields.io/pypi/v/pythonperlin?style=plastic)](https://pypi.org/project/pythonperlin/)
[![License](https://img.shields.io/pypi/l/pythonperlin?style=plastic)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/pythonperlin/badge/?version=latest)](https://pythonperlin.readthedocs.io/en/latest/?badge=latest)

<h1><p align="left">
  <img src="https://github.com/timpyrkov/pythonperlin/blob/master/docs/media/logo.png?raw=true" alt="PythonPerlin logo" height="40" style="vertical-align: middle; margin-right: 10px;">
  <span style="font-size:2.5em; vertical-align: middle;"><b>PythonPerlin</b></span>
</p></h1>

#

Perlin noise is a type of gradient noise developed by professor <b>Ken Perlin</b> in 1983 at the Department of Computer Science at New York University (NYU).

He began working on the algorithm while contributing to Disney’s 1982 science fiction film <b>"Tron"</b>, which starred Bruce Boxleitner and Peter Jurasik — both of whom later appeared in the sci-fi TV series <b>"Babylon 5"</b> (1993–1998).

In recognition of his contribution to computer graphics, Ken Perlin received an <b>Academy Award for Technical Achievement</b> in 1997 from the Academy of Motion Picture Arts and Sciences. His technique has since become a foundational tool in the procedural generation of natural-looking textures and environments.

This documentation ([https://pythonperlin.readthedocs.io](https://pythonperlin.readthedocs.io)) provides examples and explanation of using Perlin noise for procedural generation of:
- <b>Natural textures</b>: water, clouds, fire, wood, marble 
- <b>Triangulated surface grids</b>: terrain, planets

#
# Installation
```
pip install pythonperlin
```

# Quick start
```
import pythonperlin as pp
import matplotlib.pylab as plt

# 2D grid shape to seed gradients
shape = (6,4)

# Density to fill space between gradients
dens = 32

# Generate noise
noise = pp.perlin(shape, dens=dens, seed=0)

# Grid size = [s * dens for s in shape]
print(noise.shape)

# Show noise
plt.figure(figsize=(9,6))
plt.imshow(noise.T, cmap=plt.get_cmap("tab20b"))
plt.axis("off")
plt.show()
```
![](https://github.com/timpyrkov/pythonperlin/blob/master/docs/media/noise.png?raw=true)


# Tile
```
import numpy as np 

# Tile along x axis
noise = np.concatenate([noise] * 2, axis=0)

# Tile along y axis
noise = np.concatenate([noise] * 2, axis=1)

plt.figure(figsize=(6,4))
plt.imshow(noise.T, cmap=plt.get_cmap("tab20b"))
plt.plot([noise.shape[0] // 2] * 2, [0, noise.shape[1]], "--k", lw=4)
plt.plot([0, noise.shape[0]], [noise.shape[1] // 2] * 2, "--k", lw=4)
plt.axis("off")
plt.show()
```
![](https://github.com/timpyrkov/pythonperlin/blob/master/docs/media/noise_tile.png?raw=true)


# Octaves
```
# Add octaves
noise = pp.perlin(shape, dens=dens, octaves=4, seed=0)

# Show noise
plt.figure(figsize=(6,4))
plt.imshow(noise.T, cmap=plt.get_cmap("tab20b"))
plt.axis("off")
plt.show()
```
![](https://github.com/timpyrkov/pythonperlin/blob/master/docs/media/noise_octaves.png?raw=true)


# Domain warping
```
## Add domain warping
noise = pp.perlin(shape, dens=dens, octaves=4, warp=True, seed=0)

# Show noise
plt.figure(figsize=(6,4))
plt.imshow(noise.T, cmap=plt.get_cmap("tab20b"))
plt.axis("off")
plt.show()
```
![](https://github.com/timpyrkov/pythonperlin/blob/master/docs/media/noise_warp.png?raw=true)


# N-dimensional
```
# Shape specifies the number of nodes along each dimension
shape = (6,4,3)

# Density specifies the filling along each dimension if tuple
dens = (32,32,4)

# Generate noise
noise = pp.perlin(shape, dens=dens, seed=0)

# Grid size = [s * d for s, d in zip(shape, dens)]
print(noise.shape)

# Show noise slices along the z axis
fig, ax = plt.subplots(3, 4, figsize=(6,4))
for k in range(noise.shape[2]):
    i, j = k // 4, k % 4
    ax[i,j].imshow(noise[...,k].T, cmap=plt.get_cmap("tab20b"))
    ax[i,j].axis("off")
plt.tight_layout()
plt.show()
```
![](https://github.com/timpyrkov/pythonperlin/blob/master/docs/media/noise_frames.png?raw=true)


Note that the noise tiles seamlessly along the z axis too: 

The noise frames transfrom smoothly and the last frame transforms to the first frame.


# Documentation

[https://pythonperlin.readthedocs.io](https://pythonperlin.readthedocs.io)