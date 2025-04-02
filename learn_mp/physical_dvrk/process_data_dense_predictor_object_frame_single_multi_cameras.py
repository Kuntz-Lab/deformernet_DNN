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

parser = argparse.ArgumentParser(description=None)
parser.add_argument('--obj_category', default="None", type=str, help="object category. Ex: boxes_1kPa")
args = parser.parse_args()

def flip_x_y_axes(pc):
    pc[:, 0] *= -1
    pc[:, 1] *= -1
    return pc


# data_recording_path = f"/home/baothach/shape_servo_data/manipulation_points/single_physical_dvrk/multi_{args.obj_category}/mp_data"
data_recording_path = f"/home/baothach/shape_servo_data/rotation_extension/single_physical_dvrk/multi_{args.obj_category}_multi_cameras/data_2"
data_processed_path = f"/home/baothach/shape_servo_data/manipulation_points/single_physical_dvrk/multi_{args.obj_category}_multi_cameras/processed_seg_data_object_frame_2"
os.makedirs(data_processed_path, exist_ok=True)

start_time = timeit.default_timer() 
visualization = False

# file_names = sorted(os.listdir(data_recording_path))


count_data_pt = 0
for i in range(50, 10000):   # (0, 10000)
    if count_data_pt > 5000:
        break
    print("it:", i)
    if i % 50 == 0:
        print("current count:", i, " , time passed:", timeit.default_timer() - start_time)

    
    file_name = os.path.join(data_recording_path, "sample " + str(i) + ".pickle")

    if not os.path.isfile(file_name):
        print(f"{file_name} not found")
        continue     

    try:
        with open(file_name, 'rb') as handle:
            data = pickle.load(handle)
            
        pos = data["pos"][:,0]
        if max(abs(pos)) < 0.05:
            continue
            
        all_pcs = data["partial pcs"][0]
        all_goal_pcs = data["partial pcs"][1]
        ori_mp_pos = data["mani_point"]
        num_cams = 8    #8
        assert len(all_pcs) == num_cams and len(all_goal_pcs) == num_cams

        # print("len(all_pcs), len(all_goal_pcs):", len(all_pcs), len(all_goal_pcs))

        pcds = []
        pcds_goal = []
        offset = 0.2
        for j, (pc, pc_goal) in enumerate(zip(all_pcs, all_goal_pcs)):

            valid_pts_pc = np.where(abs(pc.max(axis=1)) < 0.5)[0]
            valid_pts_pc_goal = np.where(abs(pc_goal.max(axis=1)) < 0.5)[0]
            pc = pc[valid_pts_pc]
            pc_goal = pc_goal[valid_pts_pc_goal]
            all_pcs[j] = pc
            all_goal_pcs[j] = pc_goal

            pcds.append(pcd_ize(pc, color=[0,0,0]).translate([j * offset, 0, 0]))
            pcds_goal.append(pcd_ize(pc_goal, color=[1,0,0]).translate([j * offset, 0, 0]))
            # open3d.visualization.draw_geometries([pcds[j], pcds_goal[j]])
            # break
        open3d.visualization.draw_geometries(pcds + pcds_goal)
            

        for k in range(num_cams):   
            # if k != 6:
            #     continue 
            ### Down-sample point clouds
            pc = down_sampling(all_pcs[k]) # shape (num_points, 3)
            pc_goal = down_sampling(all_goal_pcs[k])
            


            ### Find 50 points nearest to the manipulation point
            
            neigh = NearestNeighbors(n_neighbors=50)
            neigh.fit(pc)
            _, nearest_idxs = neigh.kneighbors(ori_mp_pos.reshape(1, -1))
            mp_channel = np.zeros(pc.shape[0])
            mp_channel[nearest_idxs.flatten()] = 1  # shape (num_points,). Get value of 1 at the 50 nearest point, value of 0 elsewhere. 

            # Compute transformation to object frame
            transformation_matrix = world_to_object_frame_PCA(pc)
            # print("transformation_matrix:", transformation_matrix)

            # Transform points and manipulation positions
            pc = transform_point_cloud(pc, transformation_matrix)
            pc_goal = transform_point_cloud(pc_goal, transformation_matrix)   
            mp_pos = transform_point_cloud(ori_mp_pos.reshape(1, -1), transformation_matrix).flatten() 
            
            # if mp_pos[0] < 0:
            #     # print_color("Flip x and y axes", color="red")
            #     pc = flip_x_y_axes(pc)
            #     pc_goal = flip_x_y_axes(pc_goal)
            #     mp_pos = flip_x_y_axes(mp_pos.reshape(1, -1)).flatten()
            # # else:
            # #     continue

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

            fname = f"processed sample {i} cam {k}.pickle"
            pcs = (np.transpose(pc, (1, 0)), np.transpose(pc_goal, (1, 0)))     # pcs[0] and pcs[1] shape: (3, num_points)
            processed_data = {"partial pcs": pcs, "mp_labels": mp_channel, \
                                "mani_point": data["mani_point"], "obj_name": data["obj_name"]} 

            # with open(os.path.join(data_processed_path, fname), 'wb') as handle:
            #     pickle.dump(processed_data, handle, protocol=pickle.HIGHEST_PROTOCOL) 
            # count_data_pt += 1

    except Exception as e:
        print_color(f"Error processing file {file_name}: {e}")

