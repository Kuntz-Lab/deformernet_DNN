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
from utils.miscellaneous_utils import write_pickle_data, read_pickle_data
from utils.object_frame_utils import world_to_object_frame_PCA

parser = argparse.ArgumentParser(description=None)
parser.add_argument('--obj_category', default="None", type=str, help="object category. Ex: boxes_1kPa")
args = parser.parse_args()


# data_recording_path = f"/home/baothach/shape_servo_data/manipulation_points/single_physical_dvrk/multi_{args.obj_category}/mp_data"
data_recording_path = f"/home/baothach/shape_servo_data/rotation_extension/single_physical_dvrk/multi_{args.obj_category}/data"
data_processed_path = f"/home/baothach/shape_servo_data/manipulation_points/single_physical_dvrk/multi_{args.obj_category}/processed_seg_data_object_frame"
os.makedirs(data_processed_path, exist_ok=True)

start_time = timeit.default_timer() 
visualization = True

file_names = sorted(os.listdir(data_recording_path))



for i in range(0, 10000):   # (0, 10000)
    print("it:", i)
    if i % 50 == 0:
        print("current count:", i, " , time passed:", timeit.default_timer() - start_time)
    # else:
    #     continue
    
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
    mp_pos = data["mani_point"]
    neigh = NearestNeighbors(n_neighbors=50)
    neigh.fit(pc_resampled)
    _, nearest_idxs = neigh.kneighbors(mp_pos.reshape(1, -1))
    mp_channel = np.zeros(pc_resampled.shape[0])
    mp_channel[nearest_idxs.flatten()] = 1  # shape (num_points,). Get value of 1 at the 50 nearest point, value of 0 elsewhere. 

    # Compute transformation to object frame
    transformation_matrix = world_to_object_frame_PCA(pc_resampled)
    # print("transformation_matrix:", transformation_matrix)

    # Transform points and manipulation positions
    pc_resampled = transform_point_cloud(pc_resampled, transformation_matrix)
    pc_goal_resampled = transform_point_cloud(pc_goal_resampled, transformation_matrix)   
    mp_pos = transform_point_cloud(mp_pos.reshape(1, -1), transformation_matrix).flatten() 

    if visualization:
        coor_object = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        object_sphere = create_open3d_sphere(0.01, [0, 1, 0], [0, 0, 0])
        coor_world = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        coor_world.transform(transformation_matrix)         

        pcd = pcd_ize(pc_resampled, color=[0,0,0])
        colors = np.zeros((1024,3))
        colors[nearest_idxs.flatten()] = [0,1,0]
        pcd.colors =  open3d.utility.Vector3dVector(colors)
        pcd_goal = pcd_ize(pc_goal_resampled, color=[1,0,0])
        mani_point = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        mani_point.paint_uniform_color([0,0,1])
        # open3d.visualization.draw_geometries([pcd, mani_point.translate(tuple(mp_pos))]) 
        open3d.visualization.draw_geometries([pcd, pcd_goal, mani_point.translate(tuple(mp_pos)),
                                              coor_object])     # , object_sphere, coor_world


    # pcs = (np.transpose(pc_resampled, (1, 0)), np.transpose(pc_goal_resampled, (1, 0)))     # pcs[0] and pcs[1] shape: (3, num_points)
    # processed_data = {"partial pcs": pcs, "mp_labels": mp_channel, \
    #                    "mani_point": data["mani_point"], "obj_name": data["obj_name"]} 
    # with open(os.path.join(data_processed_path, file_names[i]), 'wb') as handle:
    #     pickle.dump(processed_data, handle, protocol=pickle.HIGHEST_PROTOCOL) 



