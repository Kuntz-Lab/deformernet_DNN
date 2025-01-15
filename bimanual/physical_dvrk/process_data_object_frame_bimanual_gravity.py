import open3d
import os
import numpy as np
import pickle5 as pickle
import timeit
import torch
import argparse
import sys
sys.path.append("../../")
from sklearn.neighbors import NearestNeighbors
from utils.point_cloud_utils import down_sampling, pcd_ize, world_to_object_frame, object_to_world_frame, find_min_ang_vec, is_homogeneous_matrix, transform_point_cloud
from utils.miscellaneous_utils import write_pickle_data, read_pickle_data, is_valid_3d_coordinate_frame
from sklearn.decomposition import PCA
import trimesh


ROBOT_Z_OFFSET = 0.25
two_robot_offset = 1.0

parser = argparse.ArgumentParser(description=None)
parser.add_argument('--obj_category', default="None", type=str, help="object category. Ex: box_10kPa")
args = parser.parse_args()


data_recording_path = f"/home/baothach/shape_servo_data/rotation_extension/bimanual_physical_dvrk/multi_{args.obj_category}/data"
data_processed_path = f"/home/baothach/shape_servo_data/rotation_extension/bimanual_physical_dvrk/multi_{args.obj_category}/processed_data_object_frame_gravity"
os.makedirs(data_processed_path, exist_ok=True)
assert len(os.listdir(data_processed_path)) == 0
start_time = timeit.default_timer() 

start_index = 0
max_len_data = 15000
vis = False

def world_to_object_frame_gravity(points, axes):
    # Create a trimesh.Trimesh object from the point cloud
    point_cloud = trimesh.points.PointCloud(points)

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

        pc = down_sampling(data["partial pcs"][0])
        pc_goal = down_sampling(data["partial pcs"][1])
        mp_pos_1 = data["mani_point"][:3] 
        mp_pos_2 = data["mani_point"][3:]   
        
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
        transformation_matrix = world_to_object_frame_gravity(pc, axes)

        # Transform points and manipulation positions
        pc = transform_point_cloud(pc, transformation_matrix)
        pc_goal = transform_point_cloud(pc_goal, transformation_matrix)
        mp_pos_1 = transform_point_cloud(mp_pos_1.reshape(1, -1), transformation_matrix).flatten()
        mp_pos_2 = transform_point_cloud(mp_pos_2.reshape(1, -1), transformation_matrix).flatten()
        
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
        
                
        if vis:
            coor_world = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
            coor_world.transform(transformation_matrix)  
            coor_object = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            # coor_object.transform(transformation_matrix)

            pcd = pcd_ize(pc)
            colors = np.zeros((1024, 3))
            colors[nearest_idxs_1.flatten()] = [1, 0, 0]
            colors[nearest_idxs_2.flatten()] = [0, 1, 0]
            pcd.colors = open3d.utility.Vector3dVector(colors)

            pcd_goal = pcd_ize(pc_goal, color=[1, 0, 0])
            
            mani_point_1_sphere = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            mani_point_1_sphere.paint_uniform_color([0, 0, 1])
            mani_point_1_sphere.translate(tuple(mp_pos_1))
            mani_point_2_sphere = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            mani_point_2_sphere.paint_uniform_color([1, 0, 0])
            mani_point_2_sphere.translate(tuple(mp_pos_2))
            
            open3d.visualization.draw_geometries(
                [pcd, mani_point_1_sphere, mani_point_2_sphere, pcd_goal, coor_world, coor_object]
                # [pcd, mani_point_1_sphere, mani_point_2_sphere, pcd_goal, coor_object]
            )

        pcs = (modified_pc, pc_goal.T)
        assert pcs[0].shape == (5, 1024) and pcs[1].shape == (3, 1024), "Invalid shape"

        processed_data = {"partial pcs": pcs, "full pcs": data["full pcs"], "pos": np.concatenate((data["pos"][0], data["pos"][1]), axis=None),
                        "rot": data["rot"], "twist": data["twist"], "mani_point": data["mani_point"], "obj_name":data["obj_name"]}
        
        with open(os.path.join(data_processed_path, "processed sample " + str(i) + ".pickle"), 'wb') as handle:
            pickle.dump(processed_data, handle, protocol=pickle.HIGHEST_PROTOCOL) 






















