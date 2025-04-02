import open3d
import os
import numpy as np
import pickle
import timeit
import sys
sys.path.append("../../")
from sklearn.neighbors import NearestNeighbors
import argparse
from utils.point_cloud_utils import down_sampling, pcd_ize, find_min_ang_vec, is_homogeneous_matrix, transform_point_cloud, create_open3d_sphere
from utils.miscellaneous_utils import write_pickle_data, read_pickle_data, print_color
from utils.object_frame_utils import world_to_object_frame_PCA
from copy import deepcopy

parser = argparse.ArgumentParser(description=None)
parser.add_argument('--obj_category', default="None", type=str, help="object category. Ex: boxes_1kPa")
args = parser.parse_args()


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


# data_recording_path = f"/home/baothach/shape_servo_data/manipulation_points/single_physical_dvrk/multi_{args.obj_category}/mp_data"
data_recording_path = f"/home/baothach/shape_servo_data/rotation_extension/single_physical_dvrk/multi_{args.obj_category}_multi_cameras/data_2"
data_processed_path = f"/home/baothach/shape_servo_data/rotation_extension/single_physical_dvrk/multi_{args.obj_category}/processed_data_object_frame_multi_cameras_2"
os.makedirs(data_processed_path, exist_ok=True)

start_time = timeit.default_timer() 
visualization = False

# file_names = sorted(os.listdir(data_recording_path))


count_data_pt = 0
for i in range(0, 10000):   # (0, 10000)
    if count_data_pt > 15000:
        break
    # print("it:", i)
    if i % 50 == 0:
        print("current count:", i, " , time passed:", timeit.default_timer() - start_time)

    
    file_name = os.path.join(data_recording_path, "sample " + str(i) + ".pickle")

    if not os.path.isfile(file_name):
        print(f"{file_name} not found")
        continue     

    try:
        with open(file_name, 'rb') as handle:
            data = pickle.load(handle)
                       
        all_pcs = data["partial pcs"][0]
        all_goal_pcs = data["partial pcs"][1]
        ori_mp_pos = data["mani_point"]
        
        # Pre-transformed robot action
        pos = data["pos"][:,0]
        rot = data["rot"]
        mat_1 = compose_4x4_homo_mat(rot, pos)  
        mat_1 = rotate_around_z(mat_1, np.pi)

        num_cams = 8    #8
        assert len(all_pcs) == num_cams and len(all_goal_pcs) == num_cams

        # print("len(all_pcs), len(all_goal_pcs):", len(all_pcs), len(all_goal_pcs))

        # pcds = []
        # pcds_goal = []
        # offset = 0.2
        # for j, (pc, pc_goal) in enumerate(zip(all_pcs, all_goal_pcs)):
        #     pcds.append(pcd_ize(pc, color=[0,0,0]).translate([j * offset, 0, 0]))
        #     pcds_goal.append(pcd_ize(pc_goal, color=[1,0,0]).translate([j * offset, 0, 0]))
        #     # open3d.visualization.draw_geometries([pcds[j], pcds_goal[j]])
        #     # break
        # open3d.visualization.draw_geometries(pcds + pcds_goal)
            

        for k in range(num_cams):   
            # if k != 6:
            #     continue 
            ### Down-sample point clouds
            pc = down_sampling(all_pcs[k]) # shape (num_points, 3)
            pc_goal = down_sampling(all_goal_pcs[k])

            # Compute transformation to object frame
            transformation_matrix = world_to_object_frame_PCA(pc)
            # print("transformation_matrix:", transformation_matrix)

            # Transform points and manipulation positions
            pc = transform_point_cloud(pc, transformation_matrix)
            pc_goal = transform_point_cloud(pc_goal, transformation_matrix)   
            mp_pos = transform_point_cloud(ori_mp_pos.reshape(1, -1), transformation_matrix).flatten() 

            # Nearest neighbors processing
            neigh = NearestNeighbors(n_neighbors=50)
            neigh.fit(pc)
            
            _, nearest_idxs = neigh.kneighbors(mp_pos.reshape(1, -1))
            mp_channel = np.zeros(pc.shape[0])
            mp_channel[nearest_idxs.flatten()] = 1     
            
            modified_pc = np.vstack([pc.T, mp_channel])  
            
            modified_world_to_object_H = deepcopy(transformation_matrix)
            modified_world_to_object_H[:3,3] = 0        
            transformed_mat_1 = compute_object_to_eef(modified_world_to_object_H, mat_1)        

            if visualization:
                coor_object = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                # object_sphere = create_open3d_sphere(0.01, [0, 1, 0], [0, 0, 0])
                # coor_world = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                # coor_world.transform(transformation_matrix)         

                pcd = pcd_ize(pc, color=[0,0,0])
                colors = np.zeros((1024,3))
                colors[nearest_idxs.flatten()] = [0,1,0]
                pcd.colors =  open3d.utility.Vector3dVector(colors)
                pcd_goal = pcd_ize(pc_goal, color=[1,0,0])
                mani_point = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
                mani_point.paint_uniform_color([0,0,1])
                # open3d.visualization.draw_geometries([pcd, mani_point.translate(tuple(mp_pos))]) 
                # open3d.visualization.draw_geometries([pcd, pcd_goal, mani_point.translate(tuple(mp_pos))])
                #                                     # coor_object])     # , object_sphere, coor_world

                open3d.visualization.draw_geometries([pcd, pcd_goal, mani_point.translate(tuple(mp_pos)),
                                                    coor_object])     # , object_sphere, coor_world

            
            pcs = (modified_pc, pc_goal.T)   
            assert pcs[0].shape == (4, 1024) and pcs[1].shape == (3, 1024)
                
            pos_object_frame = transformed_mat_1[:3,3]
            rot_object_frame = transformed_mat_1[:3,:3]
            fname = f"processed sample {i} cam {k}.pickle"
            
            processed_data = {"partial pcs": pcs, "pos": pos_object_frame, "rot": rot_object_frame}
            
            with open(os.path.join(data_processed_path, fname), 'wb') as handle:
                pickle.dump(processed_data, handle, protocol=pickle.HIGHEST_PROTOCOL) 
            count_data_pt += 1
    except Exception as e:
        print_color(f"Error processing file {file_name}: {e}")


