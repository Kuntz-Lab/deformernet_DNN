import open3d
import os
import numpy as np
import pickle5 as pickle
import timeit
import torch
import argparse
import sys
sys.path.append("../")
from sklearn.neighbors import NearestNeighbors
from utils.point_cloud_utils import down_sampling, pcd_ize
from utils.miscellaneous_utils import write_pickle_data, read_pickle_data


ROBOT_Z_OFFSET = 0.25
two_robot_offset = 1.0

parser = argparse.ArgumentParser(description=None)
parser.add_argument('--obj_category', default="None", type=str, help="object category. Ex: box_10kPa")
args = parser.parse_args()


data_recording_path = f"/home/baothach/shape_servo_data/rotation_extension/single_physical_dvrk/multi_{args.obj_category}/data"
data_processed_path = f"/home/baothach/shape_servo_data/rotation_extension/single_physical_dvrk/multi_{args.obj_category}/processed_data"
os.makedirs(data_processed_path, exist_ok=True)
start_time = timeit.default_timer() 

start_index = 0
max_len_data = 15000
vis = True

with torch.no_grad():
    for i in range(start_index, max_len_data):
        
        if i % 100 == 0:
            print(f"\nProcessing sample {i}. Time elapsed: {(timeit.default_timer() - start_time)/60:.2f} mins")
        
        file_name = os.path.join(data_recording_path, "sample " + str(i) + ".pickle")

        if not os.path.isfile(file_name):
            print(f"{file_name} not found")
            continue 

        with open(file_name, 'rb') as handle:
            data = pickle.load(handle)

        pc = down_sampling(data["partial pcs"][0]).transpose(1,0)
        pc_goal = down_sampling(data["partial pcs"][1]).transpose(1,0)  
        mp_pos_1 = data["mani_point"]


        
        neigh = NearestNeighbors(n_neighbors=50)
        neigh.fit(pc.transpose(1,0))
        
        _, nearest_idxs_1 = neigh.kneighbors(mp_pos_1.reshape(1, -1))
        mp_channel_1 = np.zeros(pc.shape[1])
        mp_channel_1[nearest_idxs_1.flatten()] = 1
        

        
        modified_pc = np.vstack([pc, mp_channel_1])# pc with 4th and 5th channel with value of 1 if near the MP point and 0 elsewhere
        
        
        assert modified_pc.shape == (4,1024) and pc_goal.shape == (3,1024)
        
        if vis:
            pcd = pcd_ize(np.array(pc.transpose(1,0)))
            colors = np.zeros((1024,3))
            colors[nearest_idxs_1.flatten()] = [1,0,0]
            pcd.colors =  open3d.utility.Vector3dVector(colors)

            pcd_goal = pcd_ize(np.array(pc_goal.transpose(1,0)), color=[1,0,0])
            
            mani_point_1_sphere = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            mani_point_1_sphere.paint_uniform_color([0,0,1])
            mani_point_1_sphere.translate(tuple(mp_pos_1))

            
            coor = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
            open3d.visualization.draw_geometries([pcd, mani_point_1_sphere, coor, pcd_goal])           


        # pcs = (modified_pc, pc_goal)
        # processed_data = {"partial pcs": pcs, "full pcs": data["full pcs"], "pos": data["pos"], "rot": data["rot"], 
        #                   "mani_point": data["mani_point"], "obj_name":data["obj_name"]}
        
        # with open(os.path.join(data_processed_path, "processed sample " + str(i) + ".pickle"), 'wb') as handle:
        #     pickle.dump(processed_data, handle, protocol=pickle.HIGHEST_PROTOCOL) 


    



















