from mpl_toolkits.mplot3d import Axes3D  # Needed! do not remove, despite
import matplotlib.pyplot as plt
import numpy as np
import homog_tform as ht


def draw_cam_frustum(ax, inverse_ext, label_string=None, aspect_ratio=4 / 3.0, scale=40, dotted=False):
    """Draw a traditional camera frustum / rectangular pyramid ting to show where
    the camera is."""
    # The NaN column is necessary because to plot a bunch of lines that aren't all
    # contiguous
    a = aspect_ratio
    points = np.array([[0, a,  a, -a, -a, np.NaN],
                       [0, 1, -1, -1,  1, np.NaN],
                       [0, 1,  1,  1,  1, np.NaN]])
    points *= scale

    # Indices to draw all the lines. Include an index of 5 to get a NaN
    # which allows us to start a new line.
    indices = [0, 1, 5, 0, 2, 5, 0, 3, 5, 0, 4, 5, 1, 2, 3, 4, 1]

    # import pdb; pdb.set_trace()
    pts_transformed = ht.apply(inverse_ext, points)
    pts_to_plot = np.asarray(pts_transformed[:, indices])
    col = "k-"  # solid line by default
    if dotted:
        col = 'k--'
    ax.plot(pts_to_plot[0, :], pts_to_plot[1, :], pts_to_plot[2, :], col)
    if label_string:
        ax.text(pts_to_plot[0, 0], pts_to_plot[0, 1], pts_to_plot[0, 2], label_string)


def draw_feature_visible_line(ax, homog_direction, inverse_ext, label_string=None, length=2000):
    """Given some 3D axes a direction for the ray, plus the extrinsics
    matrix, plot the ray in 3D"""
    coords = np.hstack((np.zeros((3, 1)), length * homog_direction.reshape(3, 1)))
    coords = ht.apply(inverse_ext, coords)
    if label_string:
        ax.plot(coords[0, :], coords[1, :], coords[2, :], 'b', label=label_string)
    else:
        ax.plot(coords[0, :], coords[1, :], coords[2, :], 'b')


def draw_axes_spikes(ax, ext, scale=50):
    s = scale
    pts = np.array([[0, s, 0, 0],
                    [0, 0, s, 0],
                    [0, 0, 0, s]])
    pts_tformed = ht.apply(ext, pts)

    for idx, col in zip([[0, 1], [0, 2], [0, 3]], ['r-', 'g-', 'b-']):
        ax.plot(pts_tformed[0, idx], pts_tformed[1, idx], pts_tformed[2, idx], col)

