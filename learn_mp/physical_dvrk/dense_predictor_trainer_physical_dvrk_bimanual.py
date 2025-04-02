
import torch
import torch.nn as nn


import torch.optim as optim
import torch.nn.functional as F
import os
import argparse
import logging
import socket
import sys
sys.path.append("../")
# from pointcloud_recon_2 import PointNetShapeServo3 as DeformerNet # original partial point cloud

# from architecture_seg import ManiPointSegment
# from architecture_seg_2 import ManiPointSegment3 as ManiPointSegment
from dense_predictor_pointconv_architecture import DensePredictor
from dataset_loader_mani_point import DensePredictorDatasetAllObjects
from itertools import product


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    num_batch = 0
    for batch_idx, sample in enumerate(train_loader):
        num_batch += 1
    
        pc = sample["pcs"][0].to(device)
        pc_goal = sample["pcs"][1].to(device)
        target = sample["label"].to(device)
        
        
        optimizer.zero_grad()
        output = model(pc, pc_goal)
        # print("output shape, target.shape:", output.shape, target.shape)

        loss_1 = F.nll_loss(output[:,:2,:], target[:,:,0])
        loss_2 = F.nll_loss(output[:,2:,:], target[:,:,1])
        loss = loss_1 + loss_2
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(sample), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    print('====> Epoch: {} Average loss: {:.6f}'.format(
              epoch, train_loss/num_batch))  
    logger.info('Train: Average loss: {:.6f}'.format(
              train_loss/num_batch))  





def test(model, device, test_loader, epoch):
    model.eval()
   
    test_loss = 0
    correct_1 = 0
    correct_2 = 0
    with torch.no_grad():
        for sample in test_loader:
           
            pc = sample["pcs"][0].to(device)
            pc_goal = sample["pcs"][1].to(device)
            target = sample["label"].to(device)
            
            output = model(pc, pc_goal)

            loss_1 = F.nll_loss(output[:,:2,:], target[:,:,0], reduction='sum').item()
            loss_2 = F.nll_loss(output[:,2:,:], target[:,:,1], reduction='sum').item()
            test_loss += loss_1 + loss_2

            pred = output[:,:2,:].argmax(dim=1, keepdim=True).squeeze()  # get the index of the max log-probability
            correct_1 += torch.sum(pred == target[:,:,0])

            pred = output[:,2:,:].argmax(dim=1, keepdim=True).squeeze()  # get the index of the max log-probability
            correct_2 += torch.sum(pred == target[:,:,1])

    test_loss /= len(test_loader.dataset)*1024
    # writer.add_scalar('test loss',test_loss, epoch)
    print('\nTest set: Average loss: {:.4f}, Accuracy 1: {}/{} ({:.0f}%), Accuracy 2: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct_1, len(test_loader.dataset)*1024,
        100. * correct_1 / (len(test_loader.dataset)*1024),
        correct_2, len(test_loader.dataset)*1024,
        100. * correct_2 / (len(test_loader.dataset)*1024)))

    logger.info('Test: Average loss: {:.6f}'.format(test_loss))
    logger.info('Test: Accuracy 1: {}/{} ({:.0f}%)'.format(correct_1, len(test_loader.dataset)*1024,
                                    100. * correct_1 / (len(test_loader.dataset)*1024)))
    logger.info('Test: Accuracy 2: {}/{} ({:.0f}%)\n'.format(correct_2, len(test_loader.dataset)*1024,
                                    100. * correct_2 / (len(test_loader.dataset)*1024)))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=None)
    args = parser.parse_args()
    args.batch_size = 100   #150

    # weight_path = f"/home/baothach/shape_servo_data/manipulation_points/bimanual_physical_dvrk/all_objects/weights/all_boxes"
    weight_path = f"/home/baothach/shape_servo_data/manipulation_points/bimanual_physical_dvrk/all_objects/weights/all_boxes_object_frame"
    os.makedirs(weight_path, exist_ok=True)

    logger = logging.getLogger(weight_path)
    logger.propagate = False    # no output to console
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(weight_path, "log.txt"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(f"Machine: {socket.gethostname()}")


    torch.manual_seed(2021)
    device = torch.device("cuda")

    dataset_path = "/home/baothach/shape_servo_data/manipulation_points/bimanual_physical_dvrk"
    prim_names = ["box"]   #["box", "cylinder", "hemis"]
    stiffnesses = ["1k", "5k", "10k"]   # ["1k", "5k", "10k"]
    object_names = [f"{prim_name}_{stiffness}Pa" for (prim_name, stiffness) in list(product(prim_names, stiffnesses))]
    dataset = DensePredictorDatasetAllObjects(dataset_path, object_names)

    train_len = round(len(dataset)*0.95)  
    test_len = round(len(dataset)*0.05)
    total_len = train_len + test_len

   
    train_dataset = torch.utils.data.Subset(dataset, range(0, train_len))
    test_dataset = torch.utils.data.Subset(dataset, range(train_len, total_len))
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    
    print("training data: ", len(train_dataset))
    print("test data: ", len(test_dataset))
    print("data path:", dataset.dataset_path)
    logger.info(f"Train len: {len(train_dataset)}")    
    logger.info(f"Test len: {len(test_dataset)}") 
    logger.info(f"Data path: {dataset.dataset_path}\n") 


    model = DensePredictor(num_classes=4).to(device)  # bimanual robot
    model.apply(weights_init)
    # model.load_state_dict(torch.load(os.path.join(weight_path, "epoch " + str(86))))
    
    num_epoch_total = 300   #200 
    scheduler_step = 150    #100 
  
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, scheduler_step, gamma=0.1)
    for epoch in range(0, num_epoch_total+1):
        logger.info(f"Epoch {epoch}")
        logger.info(f"Lr: {optimizer.param_groups[0]['lr']}")
        train(model, device, train_loader, optimizer, epoch)
        scheduler.step()
        test(model, device, test_loader, epoch)
        
        if epoch % 10 == 0:            
            torch.save(model.state_dict(), os.path.join(weight_path, "epoch " + str(epoch)))

