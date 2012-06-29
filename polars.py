import numpy as np

import homog_tform as ht



def polar_to_cart(azim_rad, incl_rad, radius):
    """Converts azim, incl and radius to x,y,z which
    is returned as 3xn matrix. Azim and incl must be
    in radians"""

    x = (radius * np.cos(azim_rad) * np.sin(incl_rad)).flatten()
    y = (radius * np.sin(azim_rad) * np.sin(incl_rad)).flatten()
    z = (radius * np.cos(incl_rad)).flatten()

    # Return in a 3xn array
    return np.vstack((x, y, z))


def vision_cart_to_polar(x, y=None, z=None):
    """"Converts 3d Cartesian in a vision style coordinate system
    (z is forward, y is down) into polars"""

    if (y is None) and (z is None):
        if x.shape[0] != 3:
            raise ValueError("x should be a 3xn array")
        x, y, z = x[0, :], x[1, :], x[2, :]
    else:
        # Passed as 3 separate args
        if not (x.shape == y.shape == z.shape):
            raise ValueError("x, y and z must have same shape if all supplied")

    radius = np.sqrt((x ** 2) + (y ** 2) + (z ** 2))
    azim = -np.arctan2(z, x) - (np.pi / 2)
    incl = np.arccos(y / radius) - (np.pi / 2)

    return {'r': radius, 'a': azim, 'i': incl}


def cart_to_polar(x, y=None, z=None):
    """Converts 3d cartesian to polar. Can either pass in a 3xn
    array as one argument, or separate x,y,z vectors."""

    if (y is None) and (z is None):
        if x.shape[0] != 3:
            raise ValueError("x should be a 3xn array")
        x, y, z = x[0, :], x[1, :], x[2, :]
    else:
        # Passed as 3 separate args
        if not (x.shape == y.shape == z.shape):
            raise ValueError("x, y and z must have same shape if all supplied")

    radius = np.sqrt((x ** 2) + (y ** 2) + (z ** 2))

    incl = np.arccos(z / radius)

    azim = np.arctan2(y, x)

    return {'r': radius, 'a': azim, 'i': incl}

# # To convert a spherical polar position into a homogeneous transform we need to know
# # a) camera centre - this is the translation formula we already have
# # b) look -at direction (this should be the centre, zero zero.)
# # c) up direction
# def relative_between_polars(a1, i1, a2, i2, r):
#     """given a pair of polar coords on the edge of the same sphere, 
#     (ie same radius)"""

#     pass