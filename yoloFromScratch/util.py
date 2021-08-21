"""helper functions for darknet"""

from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2

def predict_transform(prediction,inp_dim,anchors,num_classes,CUDA=True):
    """
    args:
    - prediction: output
    - inp_dim: input image dimension

    purpose: to remedy the problems of YOLO(detection layer)
    - attributes bouding boxes predicted by a cell are stacked one by one along each other 
    - dimensions of prediction maps are different since detections happen at different scales

    takes a detection feature map and turns it into a 2-d tensor, 
    where each row of the tensor corresponds to attributes of a bounding box, in the following order
    """

    batch_size = prediction.size(0)
    stride =  inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
    
    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)

    # dimensions of anchors are in accordance to the height and width of the net block
    # these attributes descibe the dimensions of the input image, 
    # which is larger (by a factor of stride) than the detection map
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    # sigmoid
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])

    # add the center offsets
    grid = np.arange(grid_size)
    a,b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)

    prediction[:,:,:2] += x_y_offset

    #log space transform height and the width
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors

    prediction[:,:,5: 5 + num_classes] = torch.sigmoid((prediction[:,:, 5 : 5 + num_classes]))
    prediction[:,:,:4] *= stride

    return prediction
    