import open3d
import os
import numpy as np
import pickle5 as pickle
import timeit
import torch
import argparse
import sys
import trimesh
sys.path.append("../../")
from sklearn.neighbors import NearestNeighbors
from utils.point_cloud_utils import down_sampling, pcd_ize
from utils.miscellaneous_utils import write_pickle_data, read_pickle_data

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
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

def transform_point_cloud(point_cloud, transformation_matrix):
    # Add homogeneous coordinate (4th component) of 1 to each point
    homogeneous_points = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1))))

    # Apply the transformation matrix to each point
    transformed_points = np.dot(homogeneous_points, transformation_matrix.T)

    # Remove the homogeneous coordinate (4th component) from the transformed points
    transformed_points = transformed_points[:, :3]

    return transformed_points



def object_to_world_frame(points):

    """  
    Compute 4x4 homogeneous transformation matrix to transform object frame to world frame. 
    The object frame is obtained by fitting a bounding box to the object partial-view point cloud.
    The centroid of the bbox is the the origin of the object frame.
    x, y, z axes are the orientation of the bbox.
    We then compare these computed axes against the ground-truth axes ([1,0,0], [0,1,0], [0,0,1]) and align them properly.
    For example, if the computed x-axis is [0.3,0.0,0.95], which is most similar to [0,0,1], this axis would be set to be the new z-axis.

    (Input) points: object partial-view point cloud. Shape (num_pts, 3)
    """

    # Create a trimesh.Trimesh object from the point cloud
    point_cloud = trimesh.points.PointCloud(points)

    # Compute the oriented bounding box (OBB) of the point cloud
    obb = point_cloud.bounding_box_oriented

    homo_mat = obb.primitive.transform
    axes = obb.primitive.transform[:3,:3]   # x, y, z axes concat together

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


    homo_mat[:3,:3] = np.column_stack((align_x_axis, align_y_axis, align_z_axis))

    assert is_homogeneous_matrix(homo_mat)

    return homo_mat


def world_to_object_frame(points):

    """  
    Compute 4x4 homogeneous transformation matrix to transform world frame to object frame. 
    The object frame is obtained by fitting a bounding box to the object partial-view point cloud.
    The centroid of the bbox is the the origin of the object frame.
    x, y, z axes are the orientation of the bbox.
    We then compare these computed axes against the ground-truth axes ([1,0,0], [0,1,0], [0,0,1]) and align them properly.
    For example, if the computed x-axis is [0.3,0.0,0.95], which is most similar to [0,0,1], this axis would be set to be the new z-axis.
    
    **This function is used to define a new frame for the object point cloud. Crucially, it creates the training data and defines the pc for test time.

    (Input) points: object partial-view point cloud. Shape (num_pts, 3)
    """

    # Create a trimesh.Trimesh object from the point cloud
    point_cloud = trimesh.points.PointCloud(points)

    # Compute the oriented bounding box (OBB) of the point cloud
    obb = point_cloud.bounding_box_oriented

    homo_mat = obb.primitive.transform
    axes = obb.primitive.transform[:3,:3]   # x, y, z axes concat together

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
    d_w_o_o = np.dot(-R_w_o, homo_mat[:3,3])
    
    homo_mat[:3,:3] = R_w_o
    homo_mat[:3,3] = d_w_o_o

    assert is_homogeneous_matrix(homo_mat)

    return homo_mat


# Main processing script
parser = argparse.ArgumentParser(description=None)
parser.add_argument('--obj_category', default="None", type=str, help="object category. Ex: box_10kPa")
args = parser.parse_args()

data_recording_path = f"/home/baothach/shape_servo_data/rotation_extension/bimanual_physical_dvrk/multi_{args.obj_category}/data"
data_processed_path = f"/home/baothach/shape_servo_data/rotation_extension/bimanual_physical_dvrk/multi_{args.obj_category}/processed_data_object_frame"
os.makedirs(data_processed_path, exist_ok=True)
assert len(os.listdir(data_processed_path)) == 0
start_time = timeit.default_timer()
start_index = 1
max_len_data = 15000
vis = True


with torch.no_grad():
    for i in range(start_index, max_len_data):
        if i % 100 == 0:
            print(f"\nProcessing sample {i}. Time elapsed: {(timeit.default_timer() - start_time) / 60:.2f} mins")
        
        file_name = os.path.join(data_recording_path, "sample " + str(i) + ".pickle")

        if not os.path.isfile(file_name):
            print(f"{file_name} not found")
            continue 

        with open(file_name, 'rb') as handle:
            data = pickle.load(handle)

        # Extract and downsample point clouds
        pc = down_sampling(data["partial pcs"][0])
        pc_goal = down_sampling(data["partial pcs"][1])
        mp_pos_1 = data["mani_point"][:3]
        mp_pos_2 = data["mani_point"][3:]

        # Compute transformation to object frame
        transformation_matrix = world_to_object_frame(pc)

        # Transform points and manipulation positions
        pc = transform_point_cloud(pc, transformation_matrix)
        pc_goal = transform_point_cloud(pc_goal, transformation_matrix)
        mp_pos_1 = transform_point_cloud(mp_pos_1.reshape(1, -1), transformation_matrix).flatten()
        mp_pos_2 = transform_point_cloud(mp_pos_2.reshape(1, -1), transformation_matrix).flatten()
        
        # Nearest neighbors processing
        neigh = NearestNeighbors(n_neighbors=50)
        neigh.fit(pc)
        
        _, nearest_idxs_1 = neigh.kneighbors(mp_pos_1.reshape(1, -1))
        mp_channel_1 = np.zeros(pc.shape[0])
        mp_channel_1[nearest_idxs_1.flatten()] = 1
        
        _, nearest_idxs_2 = neigh.kneighbors(mp_pos_2.reshape(1, -1))
        mp_channel_2 = np.zeros(pc.shape[0])
        mp_channel_2[nearest_idxs_2.flatten()] = 1        
        
        modified_pc = np.vstack([pc.T, mp_channel_1, mp_channel_2])
        
        assert modified_pc.shape == (5, 1024) and pc_goal.shape == (1024, 3)
        
        if vis:
            homo_mat = object_to_world_frame(pc)
            coor_object = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            coor_object.transform(homo_mat)  
            coor_world = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
            coor_world.transform(transformation_matrix)

            pcd = pcd_ize(pc)
            colors = np.zeros((1024, 3))
            colors[nearest_idxs_1.flatten()] = [1, 0, 0]
            colors[nearest_idxs_2.flatten()] = [0, 1, 0]
            pcd.colors = open3d.utility.Vector3dVector(colors)

            pcd_goal = pcd_ize(pc_goal, color=[1, 0, 0])
            
            mani_point_1_sphere = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            mani_point_1_sphere.paint_uniform_color([0, 0, 1])
            mani_point_1_sphere.translate(tuple(mp_pos_1))
            mani_point_2_sphere = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            mani_point_2_sphere.paint_uniform_color([1, 0, 0])
            mani_point_2_sphere.translate(tuple(mp_pos_2))
            
            # open3d.visualization.draw_geometries(
            #     [pcd, mani_point_1_sphere, mani_point_2_sphere, coor_world, pcd_goal, obb_o3d_mesh]
            # )
            open3d.visualization.draw_geometries(
                [pcd, mani_point_1_sphere, mani_point_2_sphere, pcd_goal, coor_object, coor_world]
            )

        pcs = (modified_pc, pc_goal.T)

        # processed_data = {"partial pcs": pcs, "full pcs": data["full pcs"], "pos": np.concatenate((data["pos"][0], data["pos"][1]), axis=None),
        #                 "rot": data["rot"], "twist": data["twist"], "mani_point": data["mani_point"], "obj_name":data["obj_name"]}
        
        # with open(os.path.join(data_processed_path, "processed sample " + str(i) + ".pickle"), 'wb') as handle:
        #     pickle.dump(processed_data, handle, protocol=pickle.HIGHEST_PROTOCOL) 