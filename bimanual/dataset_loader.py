import torch
import os
import numpy as np
import ast
import random
from torch.utils.data import Dataset
import pickle5 as pickle
import open3d
import sklearn

                       



class SingleBoxDataset(Dataset):
    """Shape servo dataset."""
    '''
    Dataset for surgical setup task
    '''

    def __init__(self, percentage = 1.0, dataset_path=None):
        """
        Args:

        """ 

        # self.dataset_path = "/home/baothach/shape_servo_data/bimanual/box/processed_data/"
        # self.dataset_path = "/home/baothach/shape_servo_data/bimanual/multi_boxes_1000Pa/processed_data"
        # self.dataset_path = "/home/baothach/shape_servo_data/bimanual/multi_cylinders_1000Pa/processed_data"
        # self.dataset_path = "/home/baothach/shape_servo_data/tube_connect/cylinder/processed_data"
        self.dataset_path = dataset_path

        self.filenames = os.listdir(self.dataset_path)
        
        if percentage != 1.0:
            self.filenames = os.listdir(self.dataset_path)[:int(percentage*len(self.filenames))]
 

    
    def load_pickle_data(self, filename):
        with open(os.path.join(self.dataset_path, filename), 'rb') as handle:
            return pickle.load(handle)            

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        
        # sample = self.load_pickle_data(self.filenames[idx])
        
        # # Bimanual
        # pc = torch.tensor(sample["partial pcs"][0]).float()   # original partial point cloud
        # pc_goal = torch.tensor(sample["partial pcs"][1]).float()      
        # position = (torch.tensor(sample["positions"])*1000).float()
        

        # sample = {"pcs": (pc, pc_goal), "positions": position}     


        # return sample    

        sample = self.load_pickle_data(self.filenames[idx])
        
        pc = torch.tensor(sample["partial pcs"][0]).float()   # original partial point cloud
        pc_goal = torch.tensor(sample["partial pcs"][1]).float()      

        # position = torch.tensor(sample["pos"]).squeeze().float()
        position = (torch.tensor(sample["pos"])*1000).squeeze().float()
        rot_mat_1 = torch.tensor(sample["rot"][0]).float()
        rot_mat_2 = torch.tensor(sample["rot"][1]).float()
        sample = {"pcs": (pc, pc_goal), "pos": position, "rot_1": rot_mat_1,
                  "rot_2": rot_mat_2}   



        return sample   

        