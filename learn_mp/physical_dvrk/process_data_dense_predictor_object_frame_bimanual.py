import open3d
import os
import numpy as np
import pickle
import timeit
import sys
sys.path.append("../../")
from utils.point_cloud_utils import down_sampling, pcd_ize, object_to_world_frame, find_min_ang_vec, is_homogeneous_matrix, transform_point_cloud, create_open3d_sphere
from utils.miscellaneous_utils import find_knn
from sklearn.neighbors import NearestNeighbors
import argparse
from utils.object_frame_utils import world_to_object_frame_PCA




parser = argparse.ArgumentParser(description=None)
parser.add_argument('--obj_category', default="None", type=str, help="object category. Ex: boxes_1kPa")
args = parser.parse_args()


# data_recording_path = f"/home/baothach/shape_servo_data/manipulation_points/bimanual_physical_dvrk/multi_{args.obj_category}/mp_data"
data_recording_path = f"/home/baothach/shape_servo_data/rotation_extension/bimanual_physical_dvrk/multi_{args.obj_category}/data"
data_processed_path = f"/home/baothach/shape_servo_data/manipulation_points/bimanual_physical_dvrk/multi_{args.obj_category}/processed_seg_data_object_frame"
os.makedirs(data_processed_path, exist_ok=True)

start_time = timeit.default_timer() 
visualization = False

file_names = sorted(os.listdir(data_recording_path))



for i in range(0, 10000):
    if i % 50 == 0:
        print("current count:", i, " , time passed:", timeit.default_timer() - start_time)
    
    file_name = os.path.join(data_recording_path, "sample " + str(i) + ".pickle")

    if not os.path.isfile(file_name):
        print(f"{file_name} not found")
        continue     
   

    with open(file_name, 'rb') as handle:
        data = pickle.load(handle)

    ### Down-sample point clouds
    pc = down_sampling(data["partial pcs"][0])  # shape (num_points, 3)
    pc_goal = down_sampling(data["partial pcs"][1])
    mp_pos_1 = data["mani_point"][:3]
    mp_pos_2 = data["mani_point"][3:]

    # Compute transformation to object frame
    transformation_matrix = world_to_object_frame_PCA(pc)

    # Transform points and manipulation positions
    pc = transform_point_cloud(pc, transformation_matrix)
    pc_goal = transform_point_cloud(pc_goal, transformation_matrix)
    mp_pos_1 = transform_point_cloud(mp_pos_1.reshape(1, -1), transformation_matrix).flatten()
    mp_pos_2 = transform_point_cloud(mp_pos_2.reshape(1, -1), transformation_matrix).flatten()    

    if mp_pos_1[0] < mp_pos_2[0]:
        mp_pos_1, mp_pos_2 = mp_pos_2, mp_pos_1

    ### Find 50 points nearest to the manipulation point
    mp_channel_1, nearest_idxs_1 = find_knn(pc, mp_pos_1, num_nn=50)
    mp_channel_2, nearest_idxs_2 = find_knn(pc, mp_pos_2, num_nn=50)
    mp_channel_combined = np.stack([mp_channel_1, mp_channel_2], axis=1)
    # print(mp_channel_combined.shape)


    if visualization:
        coor_object = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        pcd_goal = pcd_ize(pc_goal, color=[1,0,0])
        pcd = pcd_ize(pc)
        colors = np.zeros((1024,3))
        colors[nearest_idxs_1] = [0,1,0]
        colors[nearest_idxs_2] = [1,0,0]
        pcd.colors =  open3d.utility.Vector3dVector(colors)
        
        mani_point_1_sphere = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        mani_point_1_sphere.paint_uniform_color([0,0,1])
        mani_point_1_sphere.translate(tuple(mp_pos_1))
        mani_point_2_sphere = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        mani_point_2_sphere.paint_uniform_color([1,0,0])
        mani_point_2_sphere.translate(tuple(mp_pos_2))
        # open3d.visualization.draw_geometries([pcd, mani_point.translate(tuple(mp_pos))]) 
        open3d.visualization.draw_geometries([pcd, pcd_goal, 
                                              mani_point_1_sphere, mani_point_2_sphere,
                                              coor_object]) 


    pcs = (np.transpose(pc, (1, 0)), np.transpose(pc_goal, (1, 0)))     # pcs[0] and pcs[1] shape: (3, num_points)
    processed_data = {"partial pcs": pcs, "mp_labels": mp_channel_combined, \
                       "mani_point": data["mani_point"], "obj_name": data["obj_name"]} 
    with open(os.path.join(data_processed_path, file_names[i]), 'wb') as handle:
        pickle.dump(processed_data, handle, protocol=pickle.HIGHEST_PROTOCOL) 



