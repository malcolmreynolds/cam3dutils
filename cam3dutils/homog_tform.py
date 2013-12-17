import numpy as np
from scipy.linalg import norm
import cv2  # Need opencv for Rodrigues, my implementation wasn't working

"""Homogeneous 4x4 transforms"""

def rodrigues_vec_to_mtx(rot_vec):
    if rot_vec.shape != (3,):
        raise ValueError("rot_vec should be a 3 element vector")

    rot_mtx, jacobian = cv2.Rodrigues(rot_vec)
    return rot_mtx

def rodrigues_mtx_to_vec(rot_mtx):
    if rot_mtx.shape != (3, 3):
        raise ValueError("Expected a 3x3 matrix")

    rot_vec, jacobian = cv2.Rodrigues(rot_mtx)

    return rot_vec.squeeze()

def vec_to_tform(vec):
    "Converts a transform in 6 degrees of freedom to a 4x4 homogeneous transform"

    tform_upper_part = np.hstack((rodrigues_vec_to_mtx(vec[3:6]), np.matrix(vec[0:3]).T))
    # print "tform_upper =", tform_upper_part
    return np.vstack((tform_upper_part, np.array([0, 0, 0, 1])))


def tform_to_vec(tform):
    """Converts a 4x4 homogeneous transform to a 6 element vector storing translation in first
    3 elements, followed by 3 element rotation in axis/angle format"""
    translation = np.asarray(tform[0:3, 3].T).squeeze()
    rotation = rodrigues_mtx_to_vec(tform[0:3, 0:3])
    return np.hstack((translation, rotation))


def identity():
    " Identity 4x4 homogeneous transform"
    return np.eye(4)


def compose(*tforms):
    """Compose two or more transforms"""
    return reduce(lambda i, j: np.dot(i, j), tforms)



def apply(tform, points):
    # Concatenate a layer of ones to convert into homogeneous points, apply
    # the transform, then convert back into 3D cartesians and return
    points_hom = np.dot(tform, np.vstack((points,
                                          np.ones((1, points.shape[1])))))
    return np.array(points_hom[0:3, :] / np.tile(points_hom[3, :], (3, 1)))


def tform(r=np.eye(3), t=np.zeros((3, 1))):
    return np.vstack((np.hstack((r, t)), np.array([0, 0, 0, 1])))


def tform_translational_diff(t1, t2):
    "Returns L2 distance between translation vectors"
    return np.sum(np.power(t1[0:3, 3] - t2[0:3, 3], 2))


def tform_rotational_diff(t1, t2):
    r_t1, r_t2 = rodrigues_mtx_to_vec(t1[0:3, 0:3]), rodrigues_mtx_to_vec(t2[0:3, 0:3])
    return np.sum(np.power(r_t1 - r_t2, 2))


def rand_rot_tform(rscale=1):
    "transform which performs a random rotation"
    return tform(rodrigues_vec_to_mtx(rscale * np.random.randn(3)))


def rand_tform(tscale=1, rscale=1):
    return tform(rodrigues_vec_to_mtx(rscale * np.random.randn(3)), \
                 tscale * np.random.randn(3, 1))


def invert(tform):
    """Inverts a 4x4 transform"""
    return np.linalg.inv(tform)


def rel_tform_between(a, b):
    "Returns a transformation T_{a->b} which takes you from A to B"
    return np.dot(np.linalg.inv(a), b)


def rel_tform_between_vecs(a_v, b_v):
    """Compute the relative transform from a to b, given that both inputs and outputs
    should be provided as 6d vectors."""
    a_m, b_m = vec_to_tform(a_v), vec_to_tform(b_v)
    rel_m = rel_tform_between(a_m, b_m)
    return tform_to_vec(rel_m)


def rot(tform):
    """Get a view of the rotation matrix of a homog tform"""
    return tform[0:3, 0:3]


def trans(tform):
    """Get a view of the transformation vector of a homog tform"""
    return tform[0:3, 3]

if __name__ == "__main__":
    # Do unit tests
    import unittest

    class HTUnitTests(unittest.TestCase):
        def test_creation(self):
            self.assertTrue(np.allclose(tform(), identity()))

        def test_rodrigues(self):
            self.assertTrue(np.allclose(rodrigues_vec_to_mtx(np.array([1, 2, 3])),
                                        np.array([[-0.69492056,  0.71352099,  0.08929286],
                                                 [-0.19200697, -0.30378504,  0.93319235],
                                                 [0.69297817,    0.6313497,  0.34810748]])))
 
    unittest.main()
