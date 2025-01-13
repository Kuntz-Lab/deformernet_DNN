from numpy.core.defchararray import translate
import open3d
import os
import numpy as np
import pickle
import torch
from copy import deepcopy
import sys
sys.path.append("../")
from dense_predictor_pointconv_architecture import DensePredictor
from utils.point_cloud_utils import down_sampling, pcd_ize


device = torch.device("cuda")
model = DensePredictor(num_classes=2).to(device)
weight_path = "/home/baothach/shape_servo_data/manipulation_points/single_physical_dvrk/all_objects/weights/all_boxes"
model.load_state_dict(torch.load(os.path.join(weight_path, "epoch 200")))    
model.eval()


data_recording_path = "/home/baothach/shape_servo_data/rotation_extension/single_physical_dvrk/multi_box_1kPa/data"


file_names = sorted(os.listdir(data_recording_path))



for i in range(0, 20):    
    file_name = os.path.join(data_recording_path, file_names[i])
    print(file_name)

    with open(file_name, 'rb') as handle:
        data = pickle.load(handle)#[0]
       
    pc = down_sampling(data["partial pcs"][0])
    # pc = pc[np.random.permutation(pc.shape[0])]
    
    pc_goal = down_sampling(data["partial pcs"][1])
    # pc_goal = pc_goal[np.random.permutation(pc_goal.shape[0])]

    pc_tensor = torch.from_numpy(pc).permute(1,0).unsqueeze(0).float().to(device) 
    pc_goal_tensor = torch.from_numpy(pc_goal).permute(1,0).unsqueeze(0).float().to(device)

    gt_mp = data["mani_point"]

    pcd = pcd_ize(pc)
    pcd_goal = pcd_ize(pc_goal)


    with torch.no_grad():

        output = model(pc_tensor, pc_goal_tensor)

        success_probs = np.exp(output.squeeze().cpu().detach().numpy())[1,:]
        print(success_probs.shape)
        print(max(success_probs), min(success_probs))

        success_probs = (success_probs/max(success_probs))
        heats = np.array([[prob, 0, 0] for prob in success_probs])
        pcd.colors =  open3d.utility.Vector3dVector(heats) 

        best_mp = pc[np.argmax(success_probs)]
        best_mani_point = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        best_mani_point.paint_uniform_color([0,1,0])  
        
        mani_point = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        mani_point.paint_uniform_color([0,0,1])
  
        open3d.visualization.draw_geometries([pcd, mani_point.translate(tuple(gt_mp)), best_mani_point.translate(tuple(best_mp)), \
                                                deepcopy(pcd_goal.translate((0.2,0,0))), open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)])    




