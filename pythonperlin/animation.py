import numpy as np
import matplotlib.pyplot as plt
import imageio
import os

def perlin_noise_gif(noise, gif_path="perlin_noise.gif", cmap="viridis", duration=0.3):
    """
    Create a GIF animation from a 3D Perlin noise array, slicing along the last axis.

    Parameters
    ----------
    noise : np.ndarray
        3D array of Perlin noise (e.g., shape (nx, ny, nt))
    gif_path : str
        Output path for the GIF file
    cmap : str
        Colormap for imshow
    duration : float
        Duration of each frame in seconds
    """
    frames = []
    for i in range(noise.shape[-1]):
        fig, ax = plt.subplots()
        im = ax.imshow(noise[..., i].T, cmap=cmap, origin="lower")
        ax.axis('off')
        fig.tight_layout(pad=0)
        tmpfile = f"_tmp_frame_{i}.png"
        plt.savefig(tmpfile, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        frames.append(imageio.imread(tmpfile))
        os.remove(tmpfile)
    imageio.mimsave(gif_path, frames, duration=duration)

# Example usage:
if __name__ == "__main__":
    import pythonperlin as pp
    shape = (8, 8, 2)
    noise = pp.perlin(shape, dens=16, octaves=4, seed=1)
    perlin_noise_gif(noise, gif_path="perlin_noise.gif") 