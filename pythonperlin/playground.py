import numpy as np
import matplotlib.pyplot as plt
from worley import worley, calc_worley
from matplotlib import colormaps
from scipy.spatial import Voronoi
import io

def cobblestone_cmap(shape=(6,4), dens=64, seed=None):

    dist, centers = worley(shape, dens, seed)

    # Difference between the smallest and second smallest distance
    dist = dist[1] - dist[0]

    plt.figure(figsize=(12,6))
    plt.imshow(dist.T, cmap=plt.get_cmap("copper"))
    plt.axis("off")
    plt.show()

    return




def cobblestone_random_colormap(dist, dist_idxs, cmap_name="copper", seed=None):
    """
    Assign each cobblestone (region) a random color from the colormap,
    and use the distance as the alpha channel.
    """
    if seed is not None:
        np.random.seed(seed)
    idx_map = dist_idxs[0]
    unique_idxs = np.unique(idx_map)
    n_colors = len(unique_idxs)
    # Sample random colors from the colormap
    cmap = cm.get_cmap(cmap_name)
    color_vals = np.random.rand(n_colors)
    colors = cmap(color_vals)[:, :3]  # RGB only
    # Map each unique index to a color
    idx_to_color = {idx: colors[i] for i, idx in enumerate(unique_idxs)}
    # Build the RGB image
    rgb_img = np.zeros(idx_map.shape + (3,), dtype=float)
    for idx, color in idx_to_color.items():
        rgb_img[idx_map == idx] = color
    # Normalize dist for alpha
    alpha = (dist[0] - dist[0].min()) / (dist[0].ptp() + 1e-8)
    # Stack RGB and alpha
    rgba_img = np.concatenate([rgb_img, alpha[..., None]], axis=-1)
    return rgba_img

# Example usage:
if __name__ == "__main__":
    import pythonperlin as pp
    shape = (64, 48)
    dist, dist_idxs, centers, center_idxs = pp.calc_worley(shape, dens=8, seed=42)
    rgba_img = cobblestone_random_colormap(dist, dist_idxs, cmap_name="copper", seed=42)

    bg = cm.get_cmap("copper")[0]
    bg_img = np.ones(rgba_img.shape) * bg
    rgba_img = np.where(rgba_img == 0, bg_img, rgba_img)

    plt.imshow(rgba_img)
    plt.axis('off')
    plt.show()

def worley_starfield(shape=(20, 10), seed=None, r_disp=0.5, min_size=20, max_size=200, min_brightness=0.5, max_brightness=1.0):
    """
    Generate a starfield using Worley noise centers as star positions.

    Parameters
    ----------
    shape : tuple
        Shape of the grid for star centers.
    seed : int or None
        Random seed for reproducibility.
    r_disp : float
        Maximum random displacement from the center (in grid units).
    min_size, max_size : float
        Range of star marker sizes.
    min_brightness, max_brightness : float
        Range of star brightness (for color/alpha).

    Returns
    -------
    None (shows a matplotlib plot)
    """
    if seed is not None:
        np.random.seed(seed)

    # Get centers from Worley noise with dens=1
    _, _, centers, _ = calc_worley(shape, dens=1, seed=seed)
    x, y = centers[0].flatten(), centers[1].flatten()
    dx = x - x.astype(int)
    dy = y - y.astype(int)
    # Convert (dx, dy) to polar coordinates
    r = np.sqrt(dx**2 + dy**2)
    phi = np.arctan2(dy, dx)
    # Convert (r, phi) to Cartesian coordinates
    x = r * np.cos(phi)
    y = r * np.sin(phi)

    # Add random displacement for natural look
    phi = np.random.uniform(0, 2 * np.pi, x.shape)
    r = np.random.uniform(0, r_disp, x.shape)
    x = x + r * np.cos(phi)
    y = y + r * np.sin(phi)
    # Random star sizes and brightness
    sizes = np.random.uniform(min_size, max_size, x.shape)
    brightness = np.random.uniform(min_brightness, max_brightness, x.shape)
    colors = np.ones((len(x), 4))
    colors[:, :3] *= brightness[:, None]  # white stars, scaled by brightness
    colors[:, 3] = brightness  # alpha
    # Plot
    plt.figure(figsize=(12, 6), facecolor="black")
    plt.scatter(x, y, s=sizes, c=colors)
    plt.axis("off")
    plt.xlim(0, shape[0])
    plt.ylim(0, shape[1])
    plt.show()

def plot_voronoi_tessellation(shape=(6,4), seed=None, show_plot=True):
    """
    Create a Voronoi tessellation plot using Worley noise centers.
    
    Parameters
    ----------
    shape : tuple
        Shape of the grid for Voronoi centers.
    seed : int or None
        Random seed for reproducibility.
    show_plot : bool, default True
        Whether to display the plot immediately.
        
    Returns
    -------
    img_array : ndarray
        RGB image array of the Voronoi tessellation
    """
    # Get centers and indices from Worley noise
    _, _, centers, idxs = calc_worley(shape, dens=1, padded=True, seed=seed)
    
    # Reshape centers and idxs to (npoints, ...)-shape
    points = centers.reshape(2,-1).T
    idxs = idxs.flatten()
    
    # Generate Voronoi diagram
    vor = Voronoi(points)
    vert = vor.vertices
    edge = vor.ridge_vertices
    face = vor.regions
    
    # Create figure
    plt.figure(figsize=(12,6), facecolor="grey")
    
    # Fill faces with random colors
    rand = np.random.uniform(0.2, 0.8, len(face))
    color = plt.get_cmap("Greys")(rand)
    for i, f in enumerate(face):
        if len(f) and min(f) > 0:
            v = vert[f]
            plt.fill(v[:,0], v[:,1], c=color[i])
    
    # Draw edges
    for e in edge:
        if min(e) > 0:
            v = vert[e]
            plt.plot(v[:,0], v[:,1], c="black", lw=12)
    
    # Plot centers
    plt.scatter(*centers, c="black", s=200)
    
    # Set limits to hide periodic boundary padding points
    plt.xlim(0, shape[0])
    plt.ylim(0, shape[1])
    plt.axis("off")
    
    # Remove all axes and margins
    fig = plt.gcf()
    ax = plt.gca()
    ax.set_position([0, 0, 1, 1])  # Make axes fill the figure
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.axis('off')
    plt.margins(0, 0)
    ax.set_xticks([])
    ax.set_yticks([])

    # Save to a buffer with tight bounding box and no padding
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=False)
    buf.seek(0)
    import matplotlib.image as mpimg
    img = mpimg.imread(buf)
    buf.close()

    if show_plot:
        plt.show()
    else:
        plt.close()

    # If the image has an alpha channel, drop it for tiling
    if img.shape[2] == 4:
        img = img[:, :, :3]

    return img

def plot_voronoi_tessellation_tile(shape=(6,4), seed=None):
    """
    Create a tiled Voronoi tessellation plot (2x2 grid).
    
    Parameters
    ----------
    shape : tuple
        Shape of the grid for Voronoi centers.
    seed : int or None
        Random seed for reproducibility.
        
    Returns
    -------
    None (shows a matplotlib plot)
    """
    # Generate single tessellation
    img = plot_voronoi_tessellation(shape, seed, show_plot=False)
    
    # Create 2x2 tiled image
    tiled = np.vstack([
        np.hstack([img, img]),
        np.hstack([img, img])
    ])
    
    # Display tiled image
    plt.figure(figsize=(12,12))
    plt.imshow(tiled)
    plt.axis('off')
    plt.show()

# Example usage:
if __name__ == "__main__":
    # Example of single tessellation
    plot_voronoi_tessellation(shape=(6,4), seed=0)
    
    # Example of tiled tessellation
    plot_voronoi_tessellation_tile(shape=(6,4), seed=0)

