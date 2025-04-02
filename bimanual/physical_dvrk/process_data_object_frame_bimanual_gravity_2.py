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
from utils.point_cloud_utils import down_sampling, pcd_ize, find_min_ang_vec, is_homogeneous_matrix, transform_point_cloud, create_open3d_sphere
from utils.miscellaneous_utils import write_pickle_data, read_pickle_data
from sklearn.decomposition import PCA


def fix_signs_for_coordinate_frame(x_axis, y_axis, z_axis=np.array([0, 0, 1]), tolerance=1e-6):
    # Check orthogonality of x_axis and y_axis
    if not np.isclose(np.dot(x_axis, y_axis), 0, atol=tolerance):
        raise ValueError("Input x_axis and y_axis are not orthogonal.")

    # Check the right-hand rule and fix signs if necessary
    cross_product = np.cross(x_axis, y_axis)
    if not np.allclose(cross_product, z_axis, atol=tolerance):
        # Flip the sign of y_axis to satisfy the right-hand rule
        y_axis = -y_axis

        # Verify the new cross product aligns with z_axis
        if not np.allclose(np.cross(x_axis, y_axis), z_axis, atol=tolerance):
            raise ValueError("Failed to adjust signs to satisfy the right-hand rule.")

    return x_axis, y_axis, z_axis


def world_to_object_frame_gravity(pc):
    pca = PCA(n_components=2)
    pca.fit(pc[:, :2])
    principal_axes = pca.components_    # shape (2,2)

    x_axis = principal_axes[0]
    y_axis = principal_axes[1]
    x_axis = np.concatenate((x_axis, [0]))
    y_axis = np.concatenate((y_axis, [0]))

    z_axis = np.array([0, 0, 1])
    x_axis, y_axis, z_axis = fix_signs_for_coordinate_frame(x_axis, y_axis) # fix signs to form a valid 3D coordinate frame

    axes = np.column_stack((x_axis, y_axis, z_axis))  # shape (3,3)

    # Create a trimesh.Trimesh object from the point cloud
    point_cloud = trimesh.points.PointCloud(pc)

    # Compute the oriented bounding box (OBB) of the point cloud
    obb = point_cloud.bounding_box_oriented

    homo_mat = obb.primitive.transform
    centroid = obb.primitive.transform[:3,3] 
    d_w_o_o = -np.dot(axes.T, centroid)

    homo_mat = np.eye(4)
    homo_mat[:3,:3] = axes.T
    homo_mat[:3,3] = d_w_o_o

    assert is_homogeneous_matrix(homo_mat)
    return homo_mat


def compose_4x4_homo_mat(rotation, translation):
    ht_matrix = np.eye(4)
    ht_matrix[:3, :3] = rotation
    ht_matrix[:3, 3] = translation
    return ht_matrix

def rotate_around_z(ht_matrix, angle):
    rotation_z = np.array([
        [np.cos(angle), -np.sin(angle), 0, 0],
        [np.sin(angle),  np.cos(angle), 0, 0],
        [0,              0,             1, 0],
        [0,              0,             0, 1]
    ])
    return np.dot(rotation_z, ht_matrix)


def compute_object_to_eef(T_world_to_object, T_world_to_eef):
    """
    Computes the transformation matrix of the robot end effector in the object frame.

    Args:
        T_world_to_object (numpy.ndarray): 4x4 matrix transforming world frame to object frame.
        T_world_to_eef (numpy.ndarray): 4x4 matrix representing end effector pose in world frame.

    Returns:
        numpy.ndarray: 4x4 transformation matrix representing the end effector pose in the object frame.
    """
    # Compute the inverse of T_world_to_object
    R = T_world_to_object[:3, :3]  # Extract the 3x3 rotation matrix
    t = T_world_to_object[:3, 3]   # Extract the 3x1 translation vector

    R_inv = R.T  # Transpose of the rotation matrix
    t_inv = -np.dot(R_inv, t)  # Inverse translation

    # Construct the inverse transformation matrix
    T_object_to_world = np.eye(4)
    T_object_to_world[:3, :3] = R_inv
    T_object_to_world[:3, 3] = t_inv

    # Compute T_object_to_eef
    T_object_to_eef = np.dot(T_object_to_world, T_world_to_eef)
    
    return T_object_to_eef


# Main processing script
parser = argparse.ArgumentParser(description=None)
parser.add_argument('--obj_category', default="None", type=str, help="object category. Ex: box_10kPa")
args = parser.parse_args()

data_recording_path = f"/home/baothach/shape_servo_data/rotation_extension/bimanual_physical_dvrk/multi_{args.obj_category}/data"
data_processed_path = f"/home/baothach/shape_servo_data/rotation_extension/bimanual_physical_dvrk/multi_{args.obj_category}/processed_data_object_frame_6"
os.makedirs(data_processed_path, exist_ok=True)
# assert len(os.listdir(data_processed_path)) == 0
start_time = timeit.default_timer()
start_index = 0
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
        
        pos = data["pos"]
        rot = data["rot"]
        # print(f"\n=====================\n")
        # print(f"idx: {i}")
        # print("pos:")
        # print(pos)
        # print("\nrot:")
        # print(rot)
        
        mat_1 = compose_4x4_homo_mat(rot[0], pos[0][:,0])
        mat_2 = compose_4x4_homo_mat(rot[1], pos[1][:,0])
        mat_2 = rotate_around_z(mat_2, np.pi)
       

        # Compute transformation to object frame
        transformation_matrix = world_to_object_frame_gravity(pc)

        # Transform points and manipulation positions
        pc = transform_point_cloud(pc, transformation_matrix)
        pc_goal = transform_point_cloud(pc_goal, transformation_matrix)
        mp_pos_1 = transform_point_cloud(mp_pos_1.reshape(1, -1), transformation_matrix).flatten()
        mp_pos_2 = transform_point_cloud(mp_pos_2.reshape(1, -1), transformation_matrix).flatten()

        # Compute transformation matrix of the end effector in the object frame
        # mat_1 = compute_object_to_eef(transformation_matrix, mat_1)
        # mat_2 = compute_object_to_eef(transformation_matrix, mat_2)
        from copy import deepcopy
        modified_world_to_object_H = deepcopy(transformation_matrix)
        modified_world_to_object_H[:3,3] = 0        
        mat_1 = compute_object_to_eef(modified_world_to_object_H, mat_1)
        mat_2 = compute_object_to_eef(modified_world_to_object_H, mat_2)

        if vis:
            # coor_1 = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            # coor_1.transform(mat_1)
            # coor_2 = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            # coor_2.transform(mat_2)        
            # eef_1_sphere = create_open3d_sphere(0.01, [1, 0, 0], mat_1[:3,3])
            # eef_2_sphere = create_open3d_sphere(0.01, [0, 0, 1], mat_2[:3,3])

            coor_object = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
            object_sphere = create_open3d_sphere(0.01, [0, 1, 0], [0, 0, 0])
            coor_world = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            coor_world.transform(transformation_matrix)         
            
            pcd = pcd_ize(pc, color=[0,0,0])
            pcd_goal = pcd_ize(pc_goal, color=[1,0,0])
            open3d.visualization.draw_geometries([pcd, pcd_goal,
                                                coor_object, object_sphere, coor_world])

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
        
        pcs = (modified_pc, pc_goal.T)
        assert pcs[0].shape == (5, 1024) and pcs[1].shape == (3, 1024)

        pos_object_frame = np.concatenate((mat_1[:3,3], mat_2[:3,3]))
        rot_object_frame = (mat_1[:3,:3], mat_2[:3,:3])
        assert pos_object_frame.shape == (6,) and rot_object_frame[0].shape == (3, 3) and rot_object_frame[1].shape == (3, 3)

        # processed_data = {"partial pcs": pcs, "full pcs": data["full pcs"], "pos": np.concatenate((data["pos"][0], data["pos"][1]), axis=None),
        #                 "rot": data["rot"], "twist": data["twist"], "mani_point": data["mani_point"], "obj_name":data["obj_name"]}
        
        # with open(os.path.join(data_processed_path, "processed sample " + str(i) + ".pickle"), 'wb') as handle:
        #     pickle.dump(processed_data, handle, protocol=pickle.HIGHEST_PROTOCOL) 