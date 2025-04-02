#!/usr/bin/env python

from sklearn.decomposition import PCA
import numpy as np

import pdb


def is_homogeneous_matrix(matrix):
    # Check matrix shape
    if matrix.shape != (4, 4):
        return False

    # Check last row
    if not np.allclose(matrix[3, :], [0, 0, 0, 1]):
        return False

    # Check rotational part (3x3 upper-left submatrix)
    rotational_matrix = matrix[:3, :3]
    if not np.allclose(np.dot(rotational_matrix, rotational_matrix.T), np.eye(3), atol=1.e-6) or \
            not np.isclose(np.linalg.det(rotational_matrix), 1.0, atol=1.e-6):
        
        print(np.linalg.inv(rotational_matrix), "\n")
        print(rotational_matrix.T)        
        print(np.linalg.det(rotational_matrix))
        
        return False

    return True


def find_pca_axes(obj_cloud, verbose=False):
    '''
    Given a point cloud determine a valid, right-handed coordinate frame
    '''
    pca_operator = PCA(n_components=3, svd_solver='full')
    pca_operator.fit(obj_cloud)
    centroid = np.matrix(pca_operator.mean_).T
    x_axis = pca_operator.components_[0]
    y_axis = pca_operator.components_[1]
    z_axis = np.cross(x_axis,y_axis)

    if verbose:
        print('PCA centroid', centroid)
        print('x_axis', x_axis)
        print('y_axis', y_axis)
        print('z_axis', z_axis)
    return np.array([x_axis, y_axis, z_axis]), centroid

#Compute angles between two vectors, code is from:
#https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python/13849249#13849249
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0Zz
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def find_min_ang_vec(world_vec, cam_vecs):
    min_ang = float('inf')
    min_ang_idx = -1
    min_ang_vec = None
    for i in range(cam_vecs.shape[1]):
        angle = angle_between(world_vec, cam_vecs[:, i])
        larger_half_pi = False
        if angle > np.pi * 0.5:
            angle = np.pi - angle
            larger_half_pi = True
        if angle < min_ang:
            min_ang = angle
            min_ang_idx = i
            if larger_half_pi:
                min_ang_vec = -cam_vecs[:, i]
            else:
                min_ang_vec = cam_vecs[:, i]

    return min_ang_vec, min_ang_idx


def world_to_object_frame_PCA(obj_cloud, verbose=False):
    '''
    For the given object cloud, build an object frame using PCA and aligning to the
    world frame.
    Returns a transformation from world frame to object frame.
    '''

    # Use PCA to find a starting object frame/centroid.
    axes, centroid = find_pca_axes(obj_cloud, verbose)
    axes = np.matrix(np.column_stack(axes))

    # Rotation from object frame to frame.
    R_o_w = np.eye(3)
    
    #Find and align x axes.
    # x_axis = [1., 0., 0.]
    # align_x_axis, min_ang_axis_idx = find_min_ang_vec(x_axis, axes) 
    # axes = np.delete(axes, min_ang_axis_idx, axis=1)
    align_x_axis = axes[:, 0]
    R_o_w[0, 0] = align_x_axis[0, 0]
    R_o_w[1, 0] = align_x_axis[1, 0]
    R_o_w[2, 0] = align_x_axis[2, 0]

    #y axes
    # y_axis = [0., 1., 0.]
    # align_y_axis, min_ang_axis_idx = find_min_ang_vec(y_axis, axes) 
    # axes = np.delete(axes, min_ang_axis_idx, axis=1)
    align_y_axis = axes[:, 1]
    R_o_w[0, 1] = align_y_axis[0, 0]
    R_o_w[1, 1] = align_y_axis[1, 0]
    R_o_w[2, 1] = align_y_axis[2, 0]

    #z axes
    # z_axis = [0., 0., 1.]
    # align_z_axis, min_ang_axis_idx = find_min_ang_vec(z_axis, axes)
    align_z_axis = axes[:, 2]
    R_o_w[0, 2] = align_z_axis[0, 0]
    R_o_w[1, 2] = align_z_axis[1, 0]
    R_o_w[2, 2] = align_z_axis[2, 0]

    # Transpose to get rotation from world to object frame.
    R_w_o = np.transpose(R_o_w)
    d_w_o_o = np.dot(-R_w_o, centroid)
    
    # Build full transformation matrix.
    align_trans_matrix = np.eye(4)
    align_trans_matrix[:3,:3] = R_w_o
    align_trans_matrix[0,3] = d_w_o_o[0]
    align_trans_matrix[1,3] = d_w_o_o[1]
    align_trans_matrix[2,3] = d_w_o_o[2]   

    assert is_homogeneous_matrix(align_trans_matrix), "Not a homogeneous matrix." 

    return align_trans_matrix#, centroid

def world_to_object_frame_camera_algin(points):

    # Use PCA to find a starting object frame/centroid.
    axes, centroid = find_pca_axes(points)
    axes = np.matrix(np.column_stack(axes))

    # Find and align z axis
    z_axis = [0., 0., 1.]
    align_z_axis, min_ang_axis_idx = find_min_ang_vec(z_axis, axes)
    axes = np.delete(axes, min_ang_axis_idx, axis=1)

    # Find and align x axis.
    x_axis = [1., 0., 0.]
    align_x_axis, min_ang_axis_idx = find_min_ang_vec(x_axis, axes) 
    axes = np.delete(axes, min_ang_axis_idx, axis=1)

    # Find and align y axis
    y_axis = [0., 1., 0.]
    align_y_axis, min_ang_axis_idx = find_min_ang_vec(y_axis, axes) 

    R_o_w = np.column_stack((align_x_axis, align_y_axis, align_z_axis))
    
    # Transpose to get rotation from world to object frame.
    R_w_o = np.transpose(R_o_w)
    d_w_o_o = np.dot(-R_w_o, centroid).flatten()
    
    homo_mat = np.eye(4)
    homo_mat[:3,:3] = R_w_o
    homo_mat[:3,3] = d_w_o_o

    assert is_homogeneous_matrix(homo_mat)

    return homo_mat