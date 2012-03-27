import numpy as np
from scipy.linalg import norm
# Homogeneous 4x4 transforms

#see http://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
def rodrigues_vec_to_mtx(k):
    if k.shape != (3,):
        raise ValueError("vec should be a 3 element vector")
    theta = norm(k)
    if theta < np.finfo(k.dtype).eps:
        #rotation angle too small to care about
        return np.matrix(np.eye(3))

    norm_k = k/theta
    s_thet = np.sin(theta)
    c_thet = np.cos(theta)
    cross_prod_k = np.array([[         0, -norm_k[2],  norm_k[1]], 
                             [ norm_k[2],          0, -norm_k[0]],
                             [-norm_k[1],  norm_k[0],         0]])
    # This must give us a square matrix!
    kkT = np.matrix(norm_k).T * np.matrix(norm_k)
    
    R = np.eye(3) + s_thet*cross_prod_k + (1.0-c_thet)*(kkT - np.eye(3))
    return R
    
#see http://en.wikipedia.org/wiki/Axis_angle#Log_map_from_SO.283.29_to_so.283.29
def rodrigues_mtx_to_vec(m):
    if m.shape != (3,3):
        raise ValueError("Expected a 3x3 matrix")
    [u,s,v] = np.linalg.svd(m)
    # print "u="
    # print u
    # print "s="
    # print s
    # print "v="
    # print v
    r = np.dot(u,v)
    # print "r="
    # print r
    trace_r = (np.trace(r)-1) / 2.0
    theta = np.arccos(trace_r)
    
    # print "trace_r =",trace_r," theta =", theta
    
    if np.sin(theta) > 1e-4:
        normaliser = 1.0 / (2 * np.sin(theta))
        omega = np.array([r[2,1] - r[1,2],
                          r[0,2] - r[2,0],
                          r[1,0] - r[0,1]])
        return normaliser * omega * theta
    
    # When we are here, sin(theta) is very small
    if trace_r > 0:
        return np.array([0,0,0]) # No rotation


    # print "doing dodgy bit on r="
    # print r
    # This is all taken from bouguet - apparently written by Mike Burl
    hashvec = np.array([0, -1, -3, -9, 9, 3, 1, 13, 5, -7, -11])
    Smat = np.array([[1,1,1], [1,0,-1], [0,1,-1], [1,-1,0],
                     [1,1,0], [0,1,1], [1,0,1], [1,1,1],
                     [1,1,-1], [1,-1,-1],  [1,-1,1]]);
                     
    M = (m+np.eye(3))/2.0;
    # print "m = "
    # print M
    uabs = np.sqrt(M[0,0]);
    vabs = np.sqrt(M[1,1]);
    wabs = np.sqrt(M[2,2]);

    mvec = np.matrix([M[0,1], M[1,2], M[0,2]]);
    # print "mvec =",mvec
    syn  = ((mvec > 1e-4) - (mvec < -1e-4)).astype(np.int64); # robust sign() function - convert from bool to int
    # print "syn =",syn
    hash_val = np.matrix(syn) * np.matrix([9, 3, 1]).T #vector product here
    # print "hash = ", hash_val
    idx = np.nonzero(hash_val[0] == hashvec);
    svec = np.asarray(Smat[idx[0],:]);
    # print "svec = ", svec

    tmp = np.array([uabs, vabs, wabs])
    # print "tmp = ",tmp

    return (theta * tmp * svec).squeeze();

    # 
    # 
    # # Check bouguet for some serious witchcraft
    # print "m =", m
    # print "trace_r =", trace_r
    # print "sin(theta) =", np.sin(theta)
    # raise ValueError("not implemented yet")

def vec_to_tform(vec):
    "Converts a transform in 6 degrees of freedom to a 4x4 homogeneous transform" 
    # blah1 = rodrigues_vec_to_mtx(vec[3:6])
    # print "blah1 =", blah1
    # blah2 = np.matrix(vec[0:3]).T
    # print "blah2 =", blah2
#    return tform(rodrigues_vec_to_mtx(vec[3:6]), np.matrix(vec[]))
    tform_upper_part = np.hstack((rodrigues_vec_to_mtx(vec[3:6]), np.matrix(vec[0:3]).T))
    # print "tform_upper =", tform_upper_part
    return np.vstack((tform_upper_part, np.matrix([0,0,0,1])))
    
def tform_to_vec(tform):
    """Converts a 4x4 homogeneous transform to a 6 element vector storing translation in first 
    3 elements, followed by 3 element rotation in axis/angle format"""
    translation = np.asarray(tform[0:3,3].T).squeeze()
    rotation = rodrigues_mtx_to_vec(tform[0:3,0:3])
    return np.hstack((translation, rotation))
    
def identity():
    " Identity 4x4 homogeneous transform"
    return np.matrix(np.eye(4))

def compose(toa, tab):
    """Compose two transforms"""
    tob = toa * tab
    return tob

def apply(tform, points):
    # Concatenate a layer of ones to convert into homogeneous points, apply
    # the transform, then convert back into 3D cartesians and return
    points_hom = tform * np.vstack((points, np.ones((1, points.shape[1]))))
    return np.array(points_hom[0:3, :] / np.tile(points_hom[3, :], (3, 1)))

def tform(r=np.eye(3), t = np.zeros((3,1))):
    return np.matrix(np.vstack((np.hstack((r,t)), np.matrix([0,0,0,1]))))
    
def tform_translational_diff(t1, t2):
    "Returns L2 distance between translation vectors"
    return np.sum(np.power(t1[0:3,3] - t2[0:3,3], 2))
    
def tform_rotational_diff(t1,t2):
    r_t1, r_t2 = rodrigues_mtx_to_vec(t1[0:3,0:3]), rodrigues_mtx_to_vec(t2[0:3,0:3])
    return np.sum(np.power(r_t1 - r_t2, 2))
    
def rand_rot_tform(rscale=1):
    "transform which performs a random rotation"
    return tform(rodrigues_vec_to_mtx(rscale * np.random.randn(3)))
    
def rand_tform(tscale = 1, rscale = 1):
    return tform(rodrigues_vec_to_mtx(rscale * np.random.randn(3)), \
                 rscale * np.random.randn(3,1))

def invert_tform(tform):
    """Inverts a 4x4 transform"""
    return np.linalg.inv(tform)

def rel_tform_between(a, b):
    "Returns a transformation T_{a->b} which takes you from A to B"
    return np.linalg.inv(a) * b

def rel_tform_between_vecs(a_v, b_v):
    """Compute the relative transform from a to b, given that both inputs and outputs
    should be provided as 6d vectors."""
    a_m, b_m = vec_to_tform(a_v), vec_to_tform(b_v)
    rel_m = rel_tform_between(a_m, b_m)
    return tform_to_vec(rel_m)


    
    