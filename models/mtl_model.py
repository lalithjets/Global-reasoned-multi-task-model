'''
Project         : Global-Reasoned Multi-Task Surgical Scene Understanding
Lab             : MMLAB, National University of Singapore
contributors    : Lalithkumar Seenivasan, Sai Mitheran, Mobarakol Islam, Hongliang Ren
Note            : Code adopted and modified from Visual-Semantic Graph Attention Networks and Dual attention network for scene segmentation
'''

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

    def __init__(self, feature_encoder, scene_graph, seg_gcn_block, seg_decoder, seg_mode = None):
        super(mtl_model, self).__init__()
        self.feature_encoder = feature_encoder
        self.gcn_unit = seg_gcn_block
        self.seg_mode = seg_mode
        self.seg_decoder = seg_decoder
        self.scene_graph = scene_graph
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    def model_type1_insert(self):
        self.sg_avgpool = nn.AdaptiveAvgPool1d(1)
        self.sg_linear = nn.Linear(1040, 128)
        self.sg_feat_s1d1 = nn.Conv1d(1, 1, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

    def model_type2_insert(self):
        self.sg2_linear = nn.Linear(1040, 128)

    def model_type3_insert(self):
        # self.sf_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.sf_avgpool = nn.AdaptiveAvgPool1d(1)
        #self.sf_linear = nn.Linear(256, 128)

    def set_train_test(self, model_type):
        ''' train Feature extractor for scene graph '''
        # if model_type == 'stl-s' or model_type == 'amtl-t0' or model_type == 'amtl-t3' or model_type == 'stl-sg':
        if model_type == 'stl-s' or model_type == 'stl-sg' or model_type == 'amtl-t0' or model_type == 'amtl-t3':
            self.train_FE_SG = False
        else: 
            self.train_FE_SG = True
        
        ''' train feature extractor for segmentation '''
        # if model_type == 'stl-sg' or model_type == 'amtl-t0' or model_type == 'amtl-t3': 
        if model_type == 'stl-sg' or model_type == 'stl-sg-wfe' or model_type == 'amtl-t0' or model_type == 'amtl-t3':# or model_type == 'amtl-t1': 
            self.Train_FE_SEG = False
        else: 
            self.Train_FE_SEG = True
        
        ''' train scene graph'''
        # set train flag for scene graph
        if model_type == 'stl-s': 
            self.Train_SG = False
        else: 
            self.Train_SG = True

        ''' train segmentation GR-unit (Global-Reasoniing unit) '''
        # if model_type == 'stl-sg' or model_type == 'amtl-t0' or model_type == 'amtl-t3': 
        if model_type == 'stl-sg' or model_type == 'stl-sg-wfe' or model_type == 'amtl-t0' or model_type == 'amtl-t3': 
            self.Train_SEG_GR = False
        else: 
            self.Train_SEG_GR = True
        
        ''' train segmentation decoder '''
        # set train flag for segmentation decoder
        # if model_type == 'stl-sg' or model_type == 'amtl-t0' or model_type == 'amtl-t3': 
        if model_type == 'stl-sg' or model_type == 'stl-sg-wfe' or model_type == 'amtl-t0' or model_type == 'amtl-t3': 
            self.Train_SG_DECODER = False
        else: 
            self.Train_SG_DECODER = True

        self.model_type = model_type

    
    def forward(self, img, img_dir, det_boxes_all, node_num, spatial_feat, word2vec, roi_labels, validation=False):

        gsu_node_feat = None
        seg_inputs = None
        interaction = None
        imsize = img.size()[2:]
        
        # ====================================================== Extract node features for Scene graph ==============================================================
        if not self.train_FE_SG:
            ''' skip training the feature extractor for scene graph '''
            with torch.no_grad():
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
                    _, _, _, img_stack = self.feature_encoder(img_stack)
                    
                    img_stack = self.avgpool(img_stack)
                    img_stack = img_stack.view(img_stack.size(0), -1)

                    # # prepare graph node features
                    gsu_node_feat = img_stack if gsu_node_feat == None else torch.cat((gsu_node_feat, img_stack))
        
        else:
            # print('node_info grad enabled')
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
                _, _, _, img_stack = self.feature_encoder(img_stack)
                img_stack = self.avgpool(img_stack)
                img_stack = img_stack.view(img_stack.size(0), -1)
                # prepare graph node features
                gsu_node_feat = img_stack if gsu_node_feat == None else torch.cat((gsu_node_feat, img_stack))
        # ================================================================================================================================================================
        # ===================================================== Segmentation feature extractor ===========================================================================
        if not self.Train_FE_SEG:
            ''' Skip training feature encoder for segmentation task '''
            with torch.no_grad():
                s1, s2, s3, seg_inputs = self.feature_encoder(img)
                fe_feat = seg_inputs
        else:
            # print('segment encoder enabled')
            s1, s2, s3, seg_inputs = self.feature_encoder(img)
            fe_feat = seg_inputs
        # ================================================================================================================================================================
        # ================================================= Scene graph and segmentation GR (Global Reasoning) unit ======================================================
        if self.model_type == 'amtl-t1' or self.model_type == 'mtl-t1':
            '''
            In type 1, interaction features are passed to segmentation GR (Global Reasoning) module. 
            inside GR unit, (x = x + h + avg((x)T) * sg_feat[1x128])
            Here interation is called before GR unit.
            '''
            ''' ==== scene graph ==== '''
            # print('inside mtl-1')
            interaction, sg_feat = self.scene_graph(node_num, gsu_node_feat, spatial_feat, word2vec, roi_labels, validation= validation)
            
            ''' ==== GR (Global Reasoning) ==== '''
            edge_sum = 0
            batch_sg_feat = None
            for n in node_num:
                active_edges = n-1 if n >1 else n
                if batch_sg_feat == None:
                    batch_sg_feat = self.sg_linear(self.sg_avgpool(sg_feat[edge_sum:edge_sum+active_edges, :].unsqueeze(0).permute(0,2,1)).permute(0,2,1))
                else:
                    batch_sg_feat = torch.cat((batch_sg_feat, self.sg_linear(self.sg_avgpool(sg_feat[edge_sum:edge_sum+active_edges, :].unsqueeze(0).permute(0,2,1)).permute(0,2,1))))
                edge_sum += active_edges
            batch_sg_feat = self.sg_feat_s1d1(batch_sg_feat)
            s1, s2, s3, seg_inputs, _ = self.gcn_unit(seg_inputs, s1=s1, s2=s2, s3=s3, scene_feat = batch_sg_feat, seg_mode = self.seg_mode, model_type = self.model_type)

        elif self.model_type == 'amtl-t2' or self.model_type == 'mtl-t2':
            '''
            In type 2, interaction features are passed to segmentation GR module. Replace
            inside GR, GCN is replaced with x = x * sg_feat [128 x 128]
            Here interation is called before GR unit.
            '''
            ''' ==== scene graph ==== '''
            interaction, sg_feat = self.scene_graph(node_num, gsu_node_feat, spatial_feat, word2vec, roi_labels, validation= validation)
            
            ''' ==== GR (Global Reasoning) ==== '''
            edge_sum = 0
            batch_sg_feat = None
            for n in node_num:
                active_edges = n-1 if n >1 else n
                if batch_sg_feat == None:
                    batch_sg_feat = torch.matmul(self.sg2_linear(sg_feat[edge_sum:edge_sum+active_edges, :]).permute(1, 0), \
                                        self.sg2_linear(sg_feat[edge_sum:edge_sum+active_edges, :])).unsqueeze(0)                    
                else:
                    batch_sg_feat = torch.cat((batch_sg_feat, torch.matmul(self.sg2_linear(sg_feat[edge_sum:edge_sum+active_edges, :]).permute(1, 0), \
                                        self.sg2_linear(sg_feat[edge_sum:edge_sum+active_edges, :])).unsqueeze(0)))
                edge_sum += active_edges
            s1, s2, s3, seg_inputs, _ = self.gcn_unit(seg_inputs, s1=s1, s2=s2, s3=s3, scene_feat = batch_sg_feat, seg_mode = self.seg_mode, model_type = self.model_type)

        else:
            '''
            If it's not type 1 & 2, then GR is processed before interaction.
            '''     
            ''' ==== GR (Global Reasoning) ==== '''
            if not self.Train_SEG_GR:
                ''' skip GR unit training '''
                with torch.no_grad():
                    s1, s2, s3, seg_inputs, gi_feat = self.gcn_unit(seg_inputs, s1=s1, s2=s2, s3=s3, seg_mode = self.seg_mode, model_type = self.model_type)
            else:
                # print('segment gcn enabled')
                s1, s2, s3, seg_inputs, gi_feat = self.gcn_unit(seg_inputs, s1=s1, s2=s2, s3=s3, seg_mode = self.seg_mode, model_type = self.model_type)
                
            ''' ==== scene graph ==== '''
            if self.model_type == 'amtl-t3' or self.model_type == 'mtl-t3':
                gr_int_feat = self.sf_avgpool(gi_feat).view(gi_feat.size(0), 128)
                
                edge_sum = 0
                global_spatial_feat = None
            
                for b_i, n in enumerate(node_num):
                    active_edges = (n*(n-1)) if n >1 else n
                    if global_spatial_feat == None:
                        global_spatial_feat = torch.cat((spatial_feat[edge_sum:edge_sum+active_edges, :], gr_int_feat[b_i,:].repeat(active_edges,1)),1)
                    else:
                        global_spatial_feat = torch.cat((global_spatial_feat, torch.cat((spatial_feat[edge_sum:edge_sum+active_edges, :], gr_int_feat[b_i,:].repeat(active_edges,1)),1)))   
                    edge_sum += active_edges
                interaction, _ = self.scene_graph(node_num, gsu_node_feat, global_spatial_feat, word2vec, roi_labels, validation= validation)
            elif not self.Train_SG:
                ''' skip scene graph training '''
                with torch.no_grad():
                    global_spatial_feat = spatial_feat
                    interaction, _ = self.scene_graph(node_num, gsu_node_feat, global_spatial_feat, word2vec, roi_labels, validation= True)
            else:
                # print('interaction encoder enabled')
                global_spatial_feat = spatial_feat
                interaction, _ = self.scene_graph(node_num, gsu_node_feat, global_spatial_feat, word2vec, roi_labels, validation= validation)

        # ================================================================================================================================================================
        # ================================================= Scene graph and segmentation GR Unit =========================================================================

        ''' ============== Segmentation decoder =============='''
        if not self.Train_SG_DECODER:
            ''' skip segmentation decoder '''
            with torch.no_grad():
                seg_inputs = self.seg_decoder(seg_inputs, s1 = s1, s2 = s2, s3 =s3, imsize =  imsize, seg_mode = self.seg_mode)

        else:
            # print('segment_decoder_enabled')
            seg_inputs = self.seg_decoder(seg_inputs, s1 = s1, s2 = s2, s3 =s3, imsize =  imsize, seg_mode = self.seg_mode)
        # ================================================================================================================================================================
        
        return interaction, seg_inputs, fe_feat