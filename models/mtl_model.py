
import os
import copy
import time

import cv2
import numpy as np
from PIL import Image


import torch
import torchvision
import torch.nn as nn

class mtl_model(nn.Module):
    '''
    Multi-task model : Graph Scene Understanding and segmentation
    Forward uses features from feature_extractor
    '''

    def __init__(self, feature_encoder, scene_graph, seg_gcn_block, seg_decoder):
        super(mtl_model, self).__init__()
        self.feature_encoder = feature_encoder
        self.gcn_unit = seg_gcn_block
        self.seg_decoder = seg_decoder
        self.scene_graph = scene_graph
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    def forward(self, img, img_dir, det_boxes_all, node_num, spatial_feat, word2vec, roi_labels, validation=False):

        #fe_feature = None
        gsu_node_feat = None
        seg_inputs = None
        imsize = img.size()[2:]
        
        # feature extraction model
        for index, img_loc in enumerate(img_dir):
            _img = Image.open(img_loc).convert('RGB')
            _img = np.array(_img)
            img_stack = None
            for bndbox in det_boxes_all[index]:
                roi = np.array(bndbox).astype(int)
                roi_image = _img[roi[1]:roi[3] + 1, roi[0]:roi[2] + 1, :]
                roi_image = self.transform(cv2.resize(roi_image, (224, 224), interpolation=cv2.INTER_LINEAR))
                roi_image = torch.autograd.Variable(roi_image.unsqueeze(0))
                # stack nodes images per image
                img_stack = roi_image if img_stack == None else torch.cat((img_stack, roi_image))

            img_stack = img_stack.cuda(non_blocking=True)
            img_stack = self.feature_encoder(img_stack, f=True)
            img_stack = self.avgpool(img_stack)

            # prepare FE
            # fe_feature = img_stack if fe_feature == None else torch.cat((fe_feature, img_stack))

            # prepare graph node features
            gsu_node_feat = img_stack.view(img_stack.size(0), -1) if gsu_node_feat == None else torch.cat((gsu_node_feat, img_stack.view(img_stack.size(0), -1)))
            
        # f is False for the GCNET seg pipeline
        seg_inputs = self.feature_encoder(img, f=False)
        seg_inputs = nn.functional.interpolate(seg_inputs, size=imsize, mode='bilinear', align_corners=True)
        seg_inputs = self.gcn_unit(seg_inputs)
        seg_inputs = self.seg_decoder(seg_inputs, imsize)[0]

        # graph su model
        interaction = self.scene_graph(node_num, gsu_node_feat, spatial_feat, word2vec, roi_labels, validation= validation)
        return interaction, seg_inputs
        # return fe_feature, interaction
