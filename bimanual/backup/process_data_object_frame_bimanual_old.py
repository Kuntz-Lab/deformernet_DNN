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

# Utility functions
def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
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

    return homo_mat

def transform_point_cloud(point_cloud, transformation_matrix):
    homogeneous_points = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1))))
    transformed_points = np.dot(homogeneous_points, transformation_matrix.T)
    return transformed_points[:, :3]

# Main processing script
parser = argparse.ArgumentParser(description=None)
parser.add_argument('--obj_category', default="None", type=str, help="object category. Ex: box_10kPa")
args = parser.parse_args()

data_recording_path = f"/home/baothach/shape_servo_data/rotation_extension/bimanual_physical_dvrk/multi_{args.obj_category}/data"
start_time = timeit.default_timer()
start_index = 1
max_len_data = 2    #15000
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
        pc = down_sampling(data["partial pcs"][0]).T
        pc_goal = down_sampling(data["partial pcs"][1]).T
        mp_pos_1 = data["mani_point"][:3]
        mp_pos_2 = data["mani_point"][3:]

        # Compute transformation to object frame
        transformation_matrix = world_to_object_frame(pc.T)

        # Transform points and manipulation positions
        pc = transform_point_cloud(pc.T, transformation_matrix).T
        pc_goal = transform_point_cloud(pc_goal.T, transformation_matrix).T
        mp_pos_1 = transform_point_cloud(mp_pos_1.reshape(1, -1), transformation_matrix).flatten()
        mp_pos_2 = transform_point_cloud(mp_pos_2.reshape(1, -1), transformation_matrix).flatten()

        # Generate Trimesh Oriented Bounding Box
        trimesh_pc = trimesh.points.PointCloud(pc.T)
        obb = trimesh_pc.bounding_box_oriented

        # Extract the bounding box as a mesh
        obb_mesh = trimesh.Trimesh(vertices=obb.vertices, faces=obb.faces)

        # Convert Trimesh OBB to Open3D for visualization
        obb_o3d_mesh = open3d.geometry.TriangleMesh()
        obb_o3d_mesh.vertices = open3d.utility.Vector3dVector(obb_mesh.vertices)
        obb_o3d_mesh.triangles = open3d.utility.Vector3iVector(obb_mesh.faces)
        obb_o3d_mesh.compute_vertex_normals()
        obb_o3d_mesh.paint_uniform_color([0, 1, 0])  # Green for the bounding box

        # Transform the bounding box to the object frame
        obb_o3d_mesh.transform(transformation_matrix)

        # Transform coordinate frame to object frame
        coor = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        coor.transform(transformation_matrix)

        # Nearest neighbors processing
        neigh = NearestNeighbors(n_neighbors=50)
        neigh.fit(pc.T)
        
        _, nearest_idxs_1 = neigh.kneighbors(mp_pos_1.reshape(1, -1))
        mp_channel_1 = np.zeros(pc.shape[1])
        mp_channel_1[nearest_idxs_1.flatten()] = 1
        
        _, nearest_idxs_2 = neigh.kneighbors(mp_pos_2.reshape(1, -1))
        mp_channel_2 = np.zeros(pc.shape[1])
        mp_channel_2[nearest_idxs_2.flatten()] = 1        
        
        modified_pc = np.vstack([pc, mp_channel_1, mp_channel_2])
        
        assert modified_pc.shape == (5, 1024) and pc_goal.shape == (3, 1024)
        
        if vis:
            pcd = pcd_ize(pc.T)
            colors = np.zeros((1024, 3))
            colors[nearest_idxs_1.flatten()] = [1, 0, 0]
            colors[nearest_idxs_2.flatten()] = [0, 1, 0]
            pcd.colors = open3d.utility.Vector3dVector(colors)

            pcd_goal = pcd_ize(pc_goal.T, color=[1, 0, 0])
            
            mani_point_1_sphere = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            mani_point_1_sphere.paint_uniform_color([0, 0, 1])
            mani_point_1_sphere.translate(tuple(mp_pos_1))
            mani_point_2_sphere = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            mani_point_2_sphere.paint_uniform_color([1, 0, 0])
            mani_point_2_sphere.translate(tuple(mp_pos_2))
            
            open3d.visualization.draw_geometries(
                [pcd, mani_point_1_sphere, mani_point_2_sphere, coor, pcd_goal, obb_o3d_mesh]
            )

