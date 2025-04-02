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
import sys
sys.path.append("../")
from dense_predictor_pointconv_architecture import DensePredictor
import torch


parser = argparse.ArgumentParser(description=None)
parser.add_argument('--obj_category', default="box_1kPa", type=str, help="object category. Ex: boxes_1kPa")
args = parser.parse_args()


data_recording_path = f"/home/baothach/shape_servo_data/manipulation_points/single_physical_dvrk/multi_{args.obj_category}/processed_seg_data_object_frame"
# os.makedirs(data_processed_path, exist_ok=True)

start_time = timeit.default_timer() 
visualization = True

file_names = sorted(os.listdir(data_recording_path))

device = torch.device("cuda")
mp_seg_model = DensePredictor(num_classes=2).to(device)
weight_path = "/home/baothach/shape_servo_data/manipulation_points/single_physical_dvrk/all_objects/weights/all_boxes_object_frame"      
mp_seg_model.load_state_dict(torch.load(os.path.join(weight_path, f"epoch {300}")))   


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
    pc = data["partial pcs"][0].transpose((1,0))  # shape (num_points, 3)
    pc_goal = data["partial pcs"][1].transpose((1,0))  # shape (num_points, 3)

    # pc[:,0] = -pc[:,0]
    # pc_goal[:,0] = -pc_goal[:,0]

    mp_idxs = np.where(data["mp_labels"].astype(int) == 1)[0]

    pc_tensor = torch.tensor(pc).float().permute(1,0).unsqueeze(0).to(device)
    pc_goal_tensor = torch.tensor(pc_goal).float().permute(1,0).unsqueeze(0).to(device)
    output = mp_seg_model(pc_tensor, pc_goal_tensor)
    success_probs = np.exp(output.squeeze().cpu().detach().numpy())[1,:]
    best_mp = pc[np.argmax(success_probs)]
    print("best_mp:", best_mp)

    coor_object = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)

    gt_mani_point = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    gt_mani_point.paint_uniform_color([0,0,1])
    gt_mani_point.translate(tuple(list(best_mp)))

    pcd = pcd_ize(pc, color=[0,0,0])
    colors = np.zeros((1024,3))
    colors[mp_idxs.flatten()] = [0,1,0]
    pcd.colors =  open3d.utility.Vector3dVector(colors)
    pcd_goal = pcd_ize(pc_goal, color=[1,0,0])
    mani_point = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    open3d.visualization.draw_geometries([pcd, pcd_goal,
                                        coor_object, gt_mani_point])






