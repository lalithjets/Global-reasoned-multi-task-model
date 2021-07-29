'''
Gaussian and laplacian filters for curicullum learning
'''
import math

import torch
import torch.nn as nn


def get_gaussian_filter(kernel_size=3, sigma=2, channels=3):
    '''
    Gaussian 2D filter
    '''
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is the product of two gaussian distributions 
    # for two different variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                      torch.exp( -torch.sum((xy_grid - mean)**2., dim=-1) / (2*variance))

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    if kernel_size == 3: padding = 1
    elif kernel_size == 5: padding = 2
    else: padding = 0

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels,
                                bias=False, padding=padding)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    
    return gaussian_filter


def get_laplaceOfGaussian_filter(kernel_size=3, sigma=2, channels=3):
    '''
    laplacian 2D filter
    '''
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    mean = (kernel_size - 1)/2.

    used_sigma = sigma
    # Calculate the 2-dimensional gaussian kernel which is
    log_kernel = (-1./(math.pi*(used_sigma**4))) \
                        * (1-(torch.sum((xy_grid - mean)**2., dim=-1) / (2*(used_sigma**2)))) \
                        * torch.exp(-torch.sum((xy_grid - mean)**2., dim=-1) / (2*(used_sigma**2)))
       
    # Make sure sum of values in gaussian kernel equals 1.
    log_kernel = log_kernel / torch.sum(log_kernel)

    # Reshape to 2d depthwise convolutional weight
    log_kernel = log_kernel.view(1, 1, kernel_size, kernel_size)
    log_kernel = log_kernel.repeat(channels, 1, 1, 1)

    if kernel_size == 3: padding = 1
    elif kernel_size == 5: padding = 2
    else: padding = 0

    log_filter = nn.Conv2d( in_channels=channels, out_channels=channels, kernel_size=kernel_size, 
                            groups=channels, bias=False, padding=padding)

    log_filter.weight.data = log_kernel
    log_filter.weight.requires_grad = False
    
    return log_filter

'''
    ResNet (Pytorch implementation), together with curricullum learning filters
    Reference:
    [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        
        self.planes = planes
        self.enable_cbs = False
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut_kernel = True
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def get_new_kernels(self, fil2, fil3, kernel_size, std):
        self.enable_cbs = True
        if (fil2 == 'gau'): 
            self.kernel1 = get_gaussian_filter(kernel_size=kernel_size, sigma= std, channels=self.planes)
        elif (fil2 == 'LOG'): 
            self.kernel1 = get_laplaceOfGaussian_filter(kernel_size=kernel_size, sigma= std, channels=self.planes)

        if (fil3 == 'gau'): 
            self.kernel2 = get_gaussian_filter(kernel_size=kernel_size, sigma= std, channels=self.planes)
        elif (fil3 == 'LOG'): 
            self.kernel2 = get_laplaceOfGaussian_filter(kernel_size=kernel_size, sigma= std, channels=self.planes)

    def forward(self, x):
        out = self.conv1(x)
        
        if self.enable_cbs: out = F.relu(self.bn1(self.kernel1(out)))         
        else: out = F.relu(self.bn1(out))         
        
        out = self.conv2(out)
        
        if self.enable_cbs: out = self.bn2(self.kernel2(out))
        else: out = self.bn2(out)

        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, args):
               
        super(ResNet, self).__init__()
        self.in_planes = 64
        
        # CBS
        self.enable_cbs = args.use_cbs
        self.std = args.std
        self.factor = args.std_factor
        self.epoch = args.cbs_epoch
        self.kernel_size = args.kernel_size
        self.fil1 = args.fil1
        self.fil2 = args.fil2
        self.fil3 = args.fil3

        # Super contrast
        self.enable_SC = args.use_SC

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        if not self.enable_SC:
            self.linear = nn.Linear(512*block.expansion, args.num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        if self.enable_cbs: out = F.relu(self.bn1(self.kernel1(out)))
        else: out = F.relu(self.bn1(out))
            
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        if not self.enable_SC:
            out = self.linear(out)
        return out


    def get_new_kernels(self, epoch_count):
        if epoch_count % self.epoch == 0 and epoch_count is not 0:
            self.std *= self.factor
        if (self.fil1 == 'gau'): 
            self.kernel1 = get_gaussian_filter(kernel_size=self.kernel_size, sigma= self.std, channels=64)
        elif (self.fil1 == 'LOG'): 
            self.kernel1 = get_laplaceOfGaussian_filter(kernel_size=self.kernel_size, sigma= self.std, channels=64)

        for child in self.layer1.children():
            child.get_new_kernels(self.fil2, self.fil3, self.kernel_size, self.std)

        for child in self.layer2.children():
            child.get_new_kernels(self.fil2, self.fil3, self.kernel_size, self.std)

        for child in self.layer3.children():
            child.get_new_kernels(self.fil2, self.fil3, self.kernel_size, self.std)

        for child in self.layer4.children():
            child.get_new_kernels(self.fil2, self.fil3, self.kernel_size, self.std)



def ResNet18(args): return ResNet(BasicBlock, [2,2,2,2], args)

model_dict = {
    'resnet18': [ResNet18, 512],
    #'resnet34': [resnet34, 512],
    #'resnet50': [resnet50, 2048],
    #'resnet101': [resnet101, 2048],
}

class SupConResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, args, name='resnet18', head='mlp', feat_dim=128):
        super(SupConResNet, self).__init__()
        enc_model, dim_in = model_dict[name]
        self.encoder = enc_model(args)
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat
    
# def ResNet34(args): return ResNet(BasicBlock, [3,4,6,3], args)
# def ResNet50(args): return ResNet(Bottleneck, [3,4,6,3], args)
# def ResNet101(args):return ResNet(Bottleneck, [3,4,23,3], args)

###### spatial features extractor
'''
    stand-alone extract spatial features
    size = node x (node-1), 16 (5 + 5 + 4 + 2)
'''
import numpy as np
def center_offset(box1, box2, im_wh):
    '''
    '''
    c1 = [(box1[2]+box1[0])/2, (box1[3]+box1[1])/2]
    c2 = [(box2[2]+box2[0])/2, (box2[3]+box2[1])/2]
    offset = np.array(c1)-np.array(c2)/np.array(im_wh)
    return offset

def box_with_respect_to_img(box, im_wh):
    '''
        To get [x1/W, y1/H, x2/W, y2/H, A_box/A_img]
    '''
    # ipdb.set_trace()
    feats = [box[0]/(im_wh[0]+ 1e-6), box[1]/(im_wh[1]+ 1e-6), box[2]/(im_wh[0]+ 1e-6), box[3]/(im_wh[1]+ 1e-6)]
    box_area = (box[2]-box[0])*(box[3]-box[1])
    img_area = im_wh[0]*im_wh[1]
    feats +=[ box_area/(img_area+ 1e-6) ]
    return feats

def box1_with_respect_to_box2(box1, box2):
    '''
    '''
    feats = [ (box1[0]-box2[0])/(box2[2]-box2[0]+1e-6),
              (box1[1]-box2[1])/(box2[3]-box2[1]+ 1e-6),
              np.log((box1[2]-box1[0])/(box2[2]-box2[0]+ 1e-6)),
              np.log((box1[3]-box1[1])/(box2[3]-box2[1]+ 1e-6))   
            ]
    return feats

def calculate_spatial_feats(det_boxes, im_wh):
    '''
    '''
    spatial_feats = []
    for i in range(det_boxes.shape[0]):
        for j in range(det_boxes.shape[0]):
            if j == i: continue
            single_feat = []
            # features 5, 5, 4, 2
            box1_wrt_img = box_with_respect_to_img(det_boxes[i], im_wh)
            box2_wrt_img = box_with_respect_to_img(det_boxes[j], im_wh)
            box1_wrt_box2 = box1_with_respect_to_box2(det_boxes[i], det_boxes[j])
            offset = center_offset(det_boxes[i], det_boxes[j], im_wh)
            
            single_feat = single_feat + box1_wrt_img + box2_wrt_img + box1_wrt_box2 + offset.tolist()
            spatial_feats.append(single_feat)
    
    spatial_feats = np.array(spatial_feats)
    return spatial_feats



# '''
# Instrument Segmentation challange dataset
# '''
# #System
# import os
# import sys
# import cv2
# import h5py
# import argparse

# import torch
# import torchvision.models

# import numpy as np
# from PIL import Image
# from glob import glob
# if sys.version_info[0] == 2: import xml.etree.cElementTree as ET
# else: import xml.etree.ElementTree as ET

# # input data and IO folder location
# mlist = [1,2,3,4,5,6,7,9,10,11,12,14,15,16]

# dir_root_gt = '/media/mmlab/data_2/mobarak/mtl_graph_and_caption/datasets/instruments18/seq_'
# xml_dir_list = []

# for i in mlist:
#     xml_dir_temp = dir_root_gt + str(i) + '/xml/'
#     seq_list_each = glob(xml_dir_temp + '/*.xml')
#     xml_dir_list = xml_dir_list + seq_list_each
    
# # global variables
# INSTRUMENT_CLASSES = ('kidney', 'bipolar_forceps', 'prograsp_forceps', 'large_needle_driver',
#                       'monopolar_curved_scissors', 'ultrasound_probe', 'suction', 'clip_applier',
#                       'stapler', 'maryland_dissector', 'spatulated_monopolar_cautery')

# ACTION_CLASSES = (  'Idle', 'Grasping', 'Retraction', 'Tissue_Manipulation', 
#                     'Tool_Manipulation', 'Cutting', 'Cauterization',
#                     'Suction', 'Looping', 'Suturing', 'Clipping', 'Staple', 'Ultrasound_Sensing')

# transform = torchvision.transforms.Compose([
#                     torchvision.transforms.ToTensor(),
#                     #torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                     #std=[0.229, 0.224, 0.225])
#                     ])

# # arguments
# parser = argparse.ArgumentParser(description='feature extractor')
# parser.add_argument('--use_cbs',            type=bool,      default=True,        help='use CBS')
# parser.add_argument('--std',                type=float,     default=1.0,         help='')
# parser.add_argument('--std_factor',         type=float,     default=0.9,         help='')
# parser.add_argument('--cbs_epoch',          type=int,       default=5,           help='')
# parser.add_argument('--kernel_size',        type=int,       default=3,           help='')
# parser.add_argument('--fil1',               type=str,       default='LOG',       help='gau, LOG')
# parser.add_argument('--fil2',               type=str,       default='gau',       help='gau, LOG')
# parser.add_argument('--fil3',               type=str,       default='gau',       help='gau, LOG')

# # SupCon ARGS
# parser.add_argument('--use_SC',             type=bool,      default=True,       help='use SuperCon')

# #parser.add_argument('--savedir',            type=str,       default='vsgat/resnet18_09_cbs_ls')
# #parser.add_argument('--num_classes',        type=int,       default=9,           help='11')
# #parser.add_argument('--modelpath',           type=str,       default='checkpoint/base/ResNet18_cbs_ls_0_012345678.pkl')

# parser.add_argument('--savedir',            type=str,       default='roi_features_mtl_base') ### feature_folder = 'roi_features_resnet18_inc_sup_cbs'  'roi_features_mtl_inc'
# parser.add_argument('--num_classes',        type=int,       default=11,           help='11')
# parser.add_argument('--modelpath',          type=str,       default='../feature_extractor/checkpoint/incremental/inc_ResNet18_SC_CBS_0_012345678.pkl') ### base model: inc_ResNet18_SC_CBS_0_012345678 ; inc_ResNet18_SC_CBS_0_012345678910.pkl 
# args = parser.parse_args(args=[])
    
# # network
# if args.use_SC: 
#     feature_network = SupConResNet(args=args)
# else: 
#     feature_network = ResNet18(args)

# # CBS
# if args.use_cbs:
#     if args.use_SC:
#         feature_network.encoder.get_new_kernels(0)
#     else:
#         feature_network.get_new_kernels(0)
        
# # gpu
# num_gpu = torch.cuda.device_count()
# if num_gpu > 0:
#     device_ids = np.arange(num_gpu).tolist()    
#     if args.use_SC:
#         feature_network.encoder = torch.nn.DataParallel(feature_network.encoder)
#         feature_network = feature_network.cuda()
#     else:
#         feature_network = nn.DataParallel(feature_network, device_ids=device_ids).cuda()
            
# # load pre-trained weights
# feature_network.load_state_dict(torch.load(args.modelpath))

# # extract the encoder layer
# if args.use_SC:
#     feature_network = feature_network.encoder
# else:
#     if args.use_cbs: feature_network = nn.Sequential(*list(feature_network.module.children())[:-2])
#     else: feature_network = nn.Sequential(*list(feature_network.module.children())[:-1])

# feature_network = feature_network.cuda()

# print(feature_network)


        
# for index, _xml_dir in  enumerate(xml_dir_list):
#     img_name = os.path.basename(xml_dir_list[index][:-4])
#     _img_dir = os.path.dirname(os.path.dirname(xml_dir_list[index])) + '/left_frames/' + img_name + '.png'
#     print(_img_dir)
#     save_data_path = os.path.join(os.path.dirname(os.path.dirname(xml_dir_list[index])),args.savedir)
#     if not os.path.exists(save_data_path):
#         os.makedirs(save_data_path)
#     #print(_img_dir)
#     #if index == 2: break 

#     _xml = ET.parse(_xml_dir).getroot()
    
#     det_classes = []
#     act_classes = []
#     #node_bbox = []
#     det_boxes_all = []
#     c_flag = False
    
#     for obj in _xml.iter('objects'):
#         # object name and interaction type
#         name = obj.find('name').text.strip()
#         interact = obj.find('interaction').text.strip()
#         det_classes.append(INSTRUMENT_CLASSES.index(str(name)))
#         act_classes.append(ACTION_CLASSES.index(str(interact)))
        
#         # bounding box
#         bndbox = []
#         bbox = obj.find('bndbox') 
#         for i, pt in enumerate(['xmin', 'ymin', 'xmax', 'ymax']):         
#             bndbox.append(int(bbox.find(pt).text))
#         det_boxes_all.append(np.array(bndbox))
        
#     if c_flag: continue
        
#     tissue_num = len(np.where(np.array(det_classes)==0)[0])
#     node_num = len(det_classes)
#     if tissue_num > 0: edges = np.cumsum(node_num - np.arange(tissue_num) -1)[-1]
#     else: edges = 0
#     #print(tissue_num, node_num, edges)

#     # parse the original data to get node labels
#     edge_labels = np.zeros((edges, len(ACTION_CLASSES)))
#     edge_index = 0
#     for tissue in range (tissue_num):
#         for obj_index in range(tissue+1, node_num):
#             #print(edge_index, ";", tissue, obj_index)
#             edge_labels[edge_index, act_classes[tissue_num+edge_index]] = 1 
#             edge_index += 1

#     ###To generate adjacent matrix and added additional bbox for edge feat extraction
#     #instrument_num = node_num - 1
#     #adj_mat = np.zeros((node_num, node_num))
#     #adj_mat[0, :] = act_classes
#     #adj_mat[:, 0] = act_classes
#     #adj_mat = adj_mat.astype(int)
#     #adj_mat[adj_mat > 0] = 1
    
#     # roi features extraction
#     # node features
#     node_features = np.zeros((node_num, 512))
#     _img = Image.open(_img_dir).convert('RGB')
#     _img = np.array(_img)
#     for idx, bndbox in enumerate(det_boxes_all):
#         roi = np.array(bndbox).astype(int)
#         roi_image = _img[roi[1]:roi[3] + 1, roi[0]:roi[2] + 1, :]
#         # plt.imshow(roi_image)
#         # plt.show()
#         roi_image = transform(cv2.resize(roi_image, (224, 224), interpolation=cv2.INTER_LINEAR))
#         roi_image = torch.autograd.Variable(roi_image.unsqueeze(0)).cuda()
#         feature = feature_network(roi_image) 
#         feature = feature.view(feature.size(0), -1)
#         #print(feature.shape)
#         node_features[idx] = feature.data.cpu().numpy()

#     # spatial_features
#     spatial_features = np.array(calculate_spatial_feats(np.array(det_boxes_all), [1024, 1280]))

#     # # save to file
#     # hdf5_file = h5py.File(os.path.join(save_data_path, '{}_features.hdf5'.format(img_name)),'w')
#     # hdf5_file.create_dataset('img_name', data=img_name)
#     # hdf5_file.create_dataset('node_num', data=node_num)
#     # hdf5_file.create_dataset('classes', data=det_classes)
#     # hdf5_file.create_dataset('boxes', data=det_boxes_all)
#     # hdf5_file.create_dataset('edge_labels', data=edge_labels)
#     # hdf5_file.create_dataset('node_features', data=node_features)
#     # hdf5_file.create_dataset('spatial_features', data=spatial_features)
#     # hdf5_file.close()
#     # print('edges', edge_labels.shape, 'node_feat', node_features.shape, 'spatial_feat', spatial_features.shape)

#     np.save(os.path.join(save_data_path, '{}_node_features'.format(img_name)), node_features)
#     # np.save(os.path.join(save_data_path, '{}_spatial_features'.format(img_name)), spatial_features)
        
# print('Done')




'''=========================================================== SGH Dataset ========================================================'''
'''
SGH dataset
'''
#System
import os
import sys
import cv2
import h5py
import argparse

import torch
import torchvision.models

import numpy as np
from PIL import Image
from glob import glob
if sys.version_info[0] == 2: import xml.etree.cElementTree as ET
else: import xml.etree.ElementTree as ET

# input data and IO folder location
mlist = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]

dir_root_gt = '../datasets/SGH_dataset_2020/'
xml_dir_list = []

for i in mlist:
    xml_dir_temp = dir_root_gt + str(i) + '/xml/'
    seq_list_each = glob(xml_dir_temp + '/*.xml')
    xml_dir_list = xml_dir_list + seq_list_each
    
# global variables
INSTRUMENT_CLASSES = ('tissue', 'bipolar_forceps', 'prograsp_forceps', 'large_needle_driver',
                      'monopolar_curved_scissors', 'ultrasound_probe', 'suction', 'clip_applier',
                      'stapler', 'maryland_dissector', 'spatulated_monopolar_cautery')

ACTION_CLASSES = (  'Idle', 'Grasping', 'Retraction', 'Tissue_Manipulation', 
                    'Tool_Manipulation', 'Cutting', 'Cauterization',
                    'Suction', 'Looping', 'Suturing', 'Clipping', 'Staple', 'Ultrasound_Sensing')

transform = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor()
                    ])

# arguments
parser = argparse.ArgumentParser(description='feature extractor')
parser.add_argument('--use_cbs',            type=bool,      default=True,        help='use CBS')
parser.add_argument('--std',                type=float,     default=1.0,         help='')
parser.add_argument('--std_factor',         type=float,     default=0.9,         help='')
parser.add_argument('--cbs_epoch',          type=int,       default=5,           help='')
parser.add_argument('--kernel_size',        type=int,       default=3,           help='')
parser.add_argument('--fil1',               type=str,       default='LOG',       help='gau, LOG')
parser.add_argument('--fil2',               type=str,       default='gau',       help='gau, LOG')
parser.add_argument('--fil3',               type=str,       default='gau',       help='gau, LOG')

# SupCon ARGS
parser.add_argument('--use_SC',             type=bool,      default=True,       help='use SuperCon')

#parser.add_argument('--savedir',            type=str,       default='vsgat/resnet18_09_cbs_ts')
#parser.add_argument('--num_classes',        type=int,       default=9,           help='11')
#parser.add_argument('--modelpath',           type=str,       default='checkpoint/base/ResNet18_cbs_ts_0_012345678.pkl')

parser.add_argument('--savedir',            type=str,       default='roi_features_mtl_base') # roi_features_mtl_base; roi_features_mtl_inc
parser.add_argument('--num_classes',        type=int,       default=11,           help='11')
parser.add_argument('--modelpath',          type=str,       default='../feature_extractor/checkpoint/incremental/inc_ResNet18_SC_CBS_0_012345678.pkl') # inc_ResNet18_SC_CBS_0_012345678.pkl; inc_ResNet18_SC_CBS_0_012345678910.pkl
args = parser.parse_args(args=[])
    
        
# network
if args.use_SC: 
    feature_network = SupConResNet(args=args)
else: 
    feature_network = ResNet18(args)

# CBS
if args.use_cbs:
    if args.use_SC:
        feature_network.encoder.get_new_kernels(0)
    else:
        feature_network.get_new_kernels(0)
        
# gpu
num_gpu = torch.cuda.device_count()
if num_gpu > 0:
    device_ids = np.arange(num_gpu).tolist()    
    if args.use_SC:
        feature_network.encoder = torch.nn.DataParallel(feature_network.encoder)
        feature_network = feature_network.cuda()
    else:
        feature_network = nn.DataParallel(feature_network, device_ids=device_ids).cuda()
            
# load pre-trained weights
feature_network.load_state_dict(torch.load(args.modelpath))

# extract the encoder layer
if args.use_SC:
    feature_network = feature_network.encoder
else:
    if args.use_cbs: feature_network = nn.Sequential(*list(feature_network.module.children())[:-2])
    else: feature_network = nn.Sequential(*list(feature_network.module.children())[:-1])

feature_network = feature_network.cuda()

print(feature_network)
        
for index, _xml_dir in  enumerate(xml_dir_list):
    img_name = os.path.basename(xml_dir_list[index][:-4])
    _img_dir = os.path.dirname(os.path.dirname(xml_dir_list[index])) + '/resized_frames/' + img_name + '.png'
    print(_img_dir)
    save_data_path = os.path.join(os.path.dirname(os.path.dirname(xml_dir_list[index])),args.savedir)
    if not os.path.exists(save_data_path):
        os.makedirs(save_data_path)
    #print(_img_dir)
    #if index == 2: break 
    
    _xml = ET.parse(_xml_dir).getroot()
    
    det_classes = []
    act_classes = []
    #node_bbox = []
    det_boxes_all = []
    c_flag = False
    
    for obj in _xml.iter('objects'):
        # object name and interaction type
        name = obj.find('name').text.strip()
        interact = obj.find('interaction').text.strip()
        det_classes.append(INSTRUMENT_CLASSES.index(str(name)))
        act_classes.append(ACTION_CLASSES.index(str(interact)))
        
        # bounding box
        bndbox = []
        bbox = obj.find('bndbox') 
        for i, pt in enumerate(['xmin', 'ymin', 'xmax', 'ymax']):         
            bndbox.append(int(bbox.find(pt).text))
        det_boxes_all.append(np.array(bndbox))
        
    if c_flag: continue
        
    tissue_num = len(np.where(np.array(det_classes)==0)[0])
    node_num = len(det_classes)
    if tissue_num > 0: edges = np.cumsum(node_num - np.arange(tissue_num) -1)[-1]
    else: edges = 0
    #print(tissue_num, node_num, edges)

    # parse the original data to get node labels
    edge_labels = np.zeros((edges, len(ACTION_CLASSES)))
    edge_index = 0
    for tissue in range (tissue_num):
        for obj_index in range(tissue+1, node_num):
            #print(edge_index, ";", tissue, obj_index)
            edge_labels[edge_index, act_classes[tissue_num+edge_index]] = 1 
            edge_index += 1

    ###To generate adjacent matrix and added additional bbox for edge feat extraction
    #instrument_num = node_num - 1
    #adj_mat = np.zeros((node_num, node_num))
    #adj_mat[0, :] = act_classes
    #adj_mat[:, 0] = act_classes
    #adj_mat = adj_mat.astype(int)
    #adj_mat[adj_mat > 0] = 1
    
    # roi features extraction
    # node features
    node_features = np.zeros((node_num, 512))
    _img = Image.open(_img_dir).convert('RGB')
    _img = np.array(_img)
    for idx, bndbox in enumerate(det_boxes_all):
        roi = np.array(bndbox).astype(int)
        roi_image = _img[roi[1]:roi[3] + 1, roi[0]:roi[2] + 1, :]
        # plt.imshow(roi_image)
        # plt.show()
        roi_image = transform(cv2.resize(roi_image, (224, 224), interpolation=cv2.INTER_LINEAR))
        roi_image = torch.autograd.Variable(roi_image.unsqueeze(0)).cuda()
        feature = feature_network(roi_image)
        feature = feature.view(feature.size(0), -1)
        #print(feature.shape)
        node_features[idx] = feature.data.cpu().numpy()

    # spatial_features
    spatial_features = np.array(calculate_spatial_feats(np.array(det_boxes_all), [1024, 1280]))

    # # save to file
    # hdf5_file = h5py.File(os.path.join(save_data_path, '{}_features.hdf5'.format(img_name)),'w')
    # hdf5_file.create_dataset('img_name', data=img_name)
    # hdf5_file.create_dataset('node_num', data=node_num)
    # hdf5_file.create_dataset('classes', data=det_classes)
    # hdf5_file.create_dataset('boxes', data=det_boxes_all)
    # hdf5_file.create_dataset('edge_labels', data=edge_labels)
    # hdf5_file.create_dataset('node_features', data=node_features)
    # hdf5_file.create_dataset('spatial_features', data=spatial_features)
    # hdf5_file.close()
    # print('edges', edge_labels.shape, 'node_feat', node_features.shape, 'spatial_feat', spatial_features.shape)

    np.save(os.path.join(save_data_path, '{}_node_features'.format(img_name)), node_features)
#     np.save(os.path.join(save_data_path, '{}_spatial_features'.format(img_name)), spatial_features)

print('Done')
        

