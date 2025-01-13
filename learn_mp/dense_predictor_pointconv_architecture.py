import torch.nn as nn
import torch
import torch.nn.functional as F
# from pointnet2_utils_groupnorm import PointNetSetAbstraction,PointNetFeaturePropagation
from pointconv_util_groupnorm_2 import PointConvDensitySetAbstraction,PointConvFeaturePropagation

class DensePredictor(nn.Module):
    
    """ 
    Architecture of the dense predictor. Predict manipulation point using segmentation network.
    """
    
    def __init__(self, num_classes, normal_channel=False):
        super(DensePredictor, self).__init__()
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel

        self.sa1 = PointConvDensitySetAbstraction(npoint=512, nsample=32, in_channel=6+additional_channel, mlp=[64], bandwidth = 0.1, group_all=False)
        self.sa2 = PointConvDensitySetAbstraction(npoint=128, nsample=64, in_channel=64 + 3, mlp=[128], bandwidth = 0.2, group_all=False)
        self.sa3 = PointConvDensitySetAbstraction(npoint=1, nsample=None, in_channel=128 + 3, mlp=[256], bandwidth = 0.4, group_all=True)
        
        self.fp3 = PointConvFeaturePropagation(in_channel=128+256, mlp=[128], bandwidth = 0.4, linear_shape=128+3)
        self.fp2 = PointConvFeaturePropagation(in_channel=64+128, mlp=[64], bandwidth = 0.2, linear_shape=64+3)
        self.fp1 = PointConvFeaturePropagation(in_channel=64+additional_channel, mlp=[64, 64], bandwidth = 0.1, linear_shape=3)
        

        self.conv1 = nn.Conv1d(128, 64, 1)
        self.bn1 = nn.GroupNorm(1, 64)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(64, num_classes, 1)
        

    def forward(self, xyz, xyz_goal):
        # Set Abstraction layers
        B,C,N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)


        if self.normal_channel:
            l0_points_g = xyz_goal
            l0_xyz_g = xyz_goal[:,:3,:]
        else:
            l0_points_g = xyz_goal
            l0_xyz_g = xyz_goal
        l1_xyz_g, l1_points_g = self.sa1(l0_xyz_g, l0_points_g)
        l2_xyz_g, l2_points_g = self.sa2(l1_xyz_g, l1_points_g)
        l3_xyz_g, l3_points_g = self.sa3(l2_xyz_g, l2_points_g) 
        
        l2_points_g = self.fp3(l2_xyz_g, l3_xyz_g, l2_points_g, l3_points_g)
        l1_points_g = self.fp2(l1_xyz_g, l2_xyz_g, l1_points_g, l2_points_g)
        l0_points_g = self.fp1(l0_xyz_g, l1_xyz_g, None, l1_points_g)

        x = torch.cat([l0_points, l0_points_g], 1)
       
        
        # FC layers
        feat =  F.relu(self.bn1(self.conv1(x)))
        x = self.drop1(feat)
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        return x



# class DensePredictorBimanual(nn.Module):
    
#     """ 
#     Architecture of the dense predictor for bimanual manipulation.
#     """
    
#     def __init__(self, num_classes, normal_channel=False):
#         super(DensePredictorBimanual, self).__init__()
#         if normal_channel:
#             additional_channel = 3
#         else:
#             additional_channel = 0
#         self.normal_channel = normal_channel

#         self.sa1 = PointConvDensitySetAbstraction(npoint=512, nsample=32, in_channel=6+additional_channel, mlp=[64], bandwidth = 0.1, group_all=False)
#         self.sa2 = PointConvDensitySetAbstraction(npoint=128, nsample=64, in_channel=64 + 3, mlp=[128], bandwidth = 0.2, group_all=False)
#         self.sa3 = PointConvDensitySetAbstraction(npoint=1, nsample=None, in_channel=128 + 3, mlp=[256], bandwidth = 0.4, group_all=True)
        
#         self.fp3 = PointConvFeaturePropagation(in_channel=128+256, mlp=[128], bandwidth = 0.4, linear_shape=128+3)
#         self.fp2 = PointConvFeaturePropagation(in_channel=64+128, mlp=[64], bandwidth = 0.2, linear_shape=64+3)
#         self.fp1 = PointConvFeaturePropagation(in_channel=64+additional_channel, mlp=[64, 64], bandwidth = 0.1, linear_shape=3)
        

#         self.conv1 = nn.Conv1d(128, 64, 1)
#         self.bn1 = nn.GroupNorm(1, 64)
#         self.drop1 = nn.Dropout(0.5)
#         self.conv2 = nn.Conv1d(64, num_classes, 1)
        

#     def forward(self, xyz, xyz_goal):
#         # Set Abstraction layers
#         B,C,N = xyz.shape
#         if self.normal_channel:
#             l0_points = xyz
#             l0_xyz = xyz[:,:3,:]
#         else:
#             l0_points = xyz
#             l0_xyz = xyz
#         l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
#         l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
#         l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

#         l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
#         l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
#         l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)


#         if self.normal_channel:
#             l0_points_g = xyz_goal
#             l0_xyz_g = xyz_goal[:,:3,:]
#         else:
#             l0_points_g = xyz_goal
#             l0_xyz_g = xyz_goal
#         l1_xyz_g, l1_points_g = self.sa1(l0_xyz_g, l0_points_g)
#         l2_xyz_g, l2_points_g = self.sa2(l1_xyz_g, l1_points_g)
#         l3_xyz_g, l3_points_g = self.sa3(l2_xyz_g, l2_points_g) 
        
#         l2_points_g = self.fp3(l2_xyz_g, l3_xyz_g, l2_points_g, l3_points_g)
#         l1_points_g = self.fp2(l1_xyz_g, l2_xyz_g, l1_points_g, l2_points_g)
#         l0_points_g = self.fp1(l0_xyz_g, l1_xyz_g, None, l1_points_g)

#         x = torch.cat([l0_points, l0_points_g], 1)
        
#         # FC layers
#         feat =  F.relu(self.bn1(self.conv1(x)))
#         x = self.drop1(feat)
#         x = self.conv2(x)
#         x = F.log_softmax(x, dim=1)

#         return x


if __name__ == '__main__':

    num_classes = 2
    device = torch.device("cuda") # cuda
    pc = torch.randn((8,3,1024)).float().to(device)
    pc_goal = torch.randn((8,3,1024)).float().to(device)
    model = DensePredictor(num_classes).to(device)
    out = model(pc, pc_goal)

    print(out.shape)
