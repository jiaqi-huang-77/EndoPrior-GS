"""EndoNeRF reader for prepared images, depth, masks and poses_bounds.npy.

Camera construction and point initialisation use scene/data_loader.py.
The supplied EndoNeRF data already use this layout; no conversion is needed.
"""

from scene.data_loader import ImageDepthDataset


class EndoNeRFDataset(ImageDepthDataset):
    """Read prepared EndoNeRF sequences using the shared image/depth loader."""
