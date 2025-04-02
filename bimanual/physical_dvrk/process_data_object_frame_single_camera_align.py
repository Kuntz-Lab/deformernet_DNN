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
from utils.object_frame_utils import world_to_object_frame_camera_algin



camera_view_matrix = np.array([
    [ 1.0,        0.0,       -0.0,        0.0       ],
    [-0.0,        0.6689648, -0.7432941,  0.0       ],
    [ 0.0,        0.7432941,  0.6689648,  0.0       ],
    [-0.0,        0.27353227, -0.45340943, 1.0     ]
])


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

data_recording_path = f"/home/baothach/shape_servo_data/diffusion_defgoalnet/data/retraction_tool_deformernet"
data_processed_path = f"/home/baothach/shape_servo_data/diffusion_defgoalnet/processed_data/retraction_tool_deformernet"
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
        
        pos = data["pos"][:,0]
        rot = data["rot"]

        # Transform point clouds from world frame to camera frame
        pc = transform_point_cloud(pc, camera_view_matrix.T)
        pc_goal = transform_point_cloud(pc_goal, camera_view_matrix.T)
        mp_pos_1 = transform_point_cloud(mp_pos_1.reshape(1, -1), camera_view_matrix.T).flatten()
        

        mat_1 = compose_4x4_homo_mat(rot, pos)  
        mat_1 = rotate_around_z(mat_1, np.pi)    

        # Compute transformation to object frame (aligned with camera frame)
        transformation_matrix = world_to_object_frame_camera_algin(pc)

        # Transform from camera frame to object frame
        pc = transform_point_cloud(pc, transformation_matrix)
        pc_goal = transform_point_cloud(pc_goal, transformation_matrix)
        mp_pos_1 = transform_point_cloud(mp_pos_1.reshape(1, -1), transformation_matrix).flatten()

        # Compute transformation matrix of the end effector in the object frame
        from copy import deepcopy
        modified_world_to_object_H = deepcopy(transformation_matrix)
        modified_world_to_object_H[:3,3] = 0        
        mat_1 = compute_object_to_eef(modified_world_to_object_H, mat_1)

        # Nearest neighbors processing
        neigh = NearestNeighbors(n_neighbors=50)
        neigh.fit(pc)
        
        _, nearest_idxs_1 = neigh.kneighbors(mp_pos_1.reshape(1, -1))
        mp_channel_1 = np.zeros(pc.shape[0])
        mp_channel_1[nearest_idxs_1.flatten()] = 1     
        
        modified_pc = np.vstack([pc.T, mp_channel_1])    

        if vis:
            mani_point_1_sphere = open3d.geometry.TriangleMesh.create_sphere(radius=0.001)
            mani_point_1_sphere.paint_uniform_color([0,0,1])
            mani_point_1_sphere.translate(tuple(mp_pos_1))



            coor_object = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02)
            object_sphere = create_open3d_sphere(0.01, [0, 1, 0], [0, 0, 0])
            coor_world = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02)
            coor_world.transform(transformation_matrix)         
            
            pcd = pcd_ize(pc, color=[0,0,0])
            colors = np.zeros((1024,3))
            colors[nearest_idxs_1.flatten()] = [1,0,0]
            pcd.colors =  open3d.utility.Vector3dVector(colors)
            pcd_goal = pcd_ize(pc_goal, color=[1,0,0])
            open3d.visualization.draw_geometries([pcd, pcd_goal,
                                                coor_object, coor_world,
                                                mani_point_1_sphere])
        
        pcs = (modified_pc, pc_goal.T)
        assert pcs[0].shape == (4, 1024) and pcs[1].shape == (3, 1024)

        pos_object_frame = mat_1[:3,3]
        rot_object_frame = mat_1[:3,:3]
        assert pos_object_frame.shape == (3,) and rot_object_frame.shape == (3, 3)

        # processed_data = {"partial pcs": pcs, "pos": pos_object_frame, "rot": rot_object_frame}
        
        # with open(os.path.join(data_processed_path, "processed sample " + str(i) + ".pickle"), 'wb') as handle:
        #     pickle.dump(processed_data, handle, protocol=pickle.HIGHEST_PROTOCOL)