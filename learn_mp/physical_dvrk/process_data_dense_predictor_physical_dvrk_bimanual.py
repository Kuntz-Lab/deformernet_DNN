import open3d
import os
import numpy as np
import pickle
import timeit
import sys
sys.path.append("../../")
from utils.point_cloud_utils import down_sampling, pcd_ize
from utils.miscellaneous_utils import find_knn
from sklearn.neighbors import NearestNeighbors
import argparse


parser = argparse.ArgumentParser(description=None)
parser.add_argument('--obj_category', default="None", type=str, help="object category. Ex: boxes_1kPa")
args = parser.parse_args()


# data_recording_path = f"/home/baothach/shape_servo_data/manipulation_points/bimanual_physical_dvrk/multi_{args.obj_category}/mp_data"
data_recording_path = f"/home/baothach/shape_servo_data/rotation_extension/bimanual_physical_dvrk/multi_{args.obj_category}/data"
data_processed_path = f"/home/baothach/shape_servo_data/manipulation_points/bimanual_physical_dvrk/multi_{args.obj_category}/processed_seg_data"
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
    pc_resampled = down_sampling(data["partial pcs"][0])  # shape (num_points, 3)
    pc_goal_resampled = down_sampling(data["partial pcs"][1])

    ### Find 50 points nearest to the manipulation point
    mp_channel_1, nearest_idxs_1 = find_knn(pc_resampled, data["mani_point"][:3], num_nn=50)
    mp_channel_2, nearest_idxs_2 = find_knn(pc_resampled, data["mani_point"][3:], num_nn=50)
    mp_channel_combined = np.stack([mp_channel_1, mp_channel_2], axis=1)
    # print(mp_channel_combined.shape)
    

    if visualization:
        pcd_goal = pcd_ize(pc_goal_resampled, color=[1,0,0])
        pcd = pcd_ize(pc_resampled)
        colors = np.zeros((1024,3))
        colors[nearest_idxs_1] = [0,1,0]
        colors[nearest_idxs_2] = [1,0,0]
        pcd.colors =  open3d.utility.Vector3dVector(colors)
        
        mani_point_1 = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        mani_point_1.paint_uniform_color([0,1,0])
        mani_point_1.translate(data["mani_point"][:3])

        mani_point_2 = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        mani_point_2.paint_uniform_color([1,0,0])
        mani_point_2.translate(data["mani_point"][3:])
        # open3d.visualization.draw_geometries([pcd, mani_point.translate(tuple(mp_pos))]) 
        open3d.visualization.draw_geometries([pcd, pcd_goal.translate((0.0,0,0)), mani_point_1, mani_point_2]) 


    pcs = (np.transpose(pc_resampled, (1, 0)), np.transpose(pc_goal_resampled, (1, 0)))     # pcs[0] and pcs[1] shape: (3, num_points)
    processed_data = {"partial pcs": pcs, "mp_labels": mp_channel_combined, \
                       "mani_point": data["mani_point"], "obj_name": data["obj_name"]} 
    with open(os.path.join(data_processed_path, file_names[i]), 'wb') as handle:
        pickle.dump(processed_data, handle, protocol=pickle.HIGHEST_PROTOCOL) 



