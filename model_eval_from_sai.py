import numpy as np
import json
from sklearn.metrics import confusion_matrix
import numpy as np
from tqdm import tqdm
import os
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
from tabulate import tabulate
import sys
sys.path.append('../')

from models.mtl_model import *
from models.scene_graph import *
from models.surgicalDataset import *
from models.segmentation_model import get_gcnet  # for the get_gcnet function

import utils.io as io
from utils.vis_tool import vis_img
from utils.scene_graph_eval_matrix import *
from utils.segmentation_eval_matrix import *  # SegmentationLoss and Eval code


def seed_everything(seed=27):
    '''
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_model(args, load_pretrained = True):
    '''
    Build MTL model
    1) Feature Extraction
    3) Graph Scene Understanding Model
    '''

    '''==== graph model ===='''
    # graph model
    scene_graph = AGRNN(bias=True, bn=False, dropout=0.3, multi_attn=False, layer=1, diff_edge=False, use_cbs=args.use_cbs)
    if args.use_cbs: scene_graph.grnn1.gnn.apply_h_h_edge.get_new_kernels(0)

    # graph load pre-trained weights
    if load_pretrained:
        pretrained_model = torch.load(args.gsu_checkpoint)
        scene_graph.load_state_dict(pretrained_model['state_dict'])
    # scene_graph.eval().eval()

    '''==== Feature extractor ===='''
    # feature extraction model
    seg_model = get_gcnet(backbone='resnet18_8s_model_cbs')
    if args.use_cbs: seg_model.pretrained.get_new_kernels(0)

    # # based on cuda
    # num_gpu = torch.cuda.device_count()
    # if num_gpu > 0:
    #     device_ids = np.arange(num_gpu).tolist()
    #     feature_encoder = nn.DataParallel(feature_encoder, device_ids=device_ids)

    # # feature extraction pre-trained weights
    # feature_encoder.load_state_dict(torch.load(args.fe_modelpath))
    # feature_encoder = feature_encoder.module

    if args.use_cbs: print("Using CBS")
    else: print("Not Using CBS")

    model = mtl_model(seg_model.pretrained, scene_graph, seg_model.gcn_block, seg_model.decoder, seg_mode=args.model)
    model.to(torch.device('cpu'))
    return model

    # /media/mobarak/data/lalith/mtl_scene_understanding_and_segmentation/checkpoints/stl_s_v1/stl_s_v1/epoch_train/checkpoint_D168_epoch.pth
    # /media/mobarak/data/lalith/mtl_scene_understanding_and_segmentation/checkpoints/stl_s_v2_gc/stl_s_v2_gc/epoch_train/checkpoint_D168_epoch.pth
    # /media/mobarak/data/lalith/mtl_scene_understanding_and_segmentation/checkpoints/stl_s/stl_s/epoch_train/checkpoint_D153_epoch.pth
    # /media/mobarak/data/lalith/mtl_scene_understanding_and_segmentation/checkpoints/stl_s_ng/stl_s_ng/epoch_train/checkpoint_D168_epoch.pth

def load(args, model):
    print('Loading pre-trained weights from segmentation STL')
    if args.model == 'v0':
        pretrained_model = torch.load('/media/mobarak/data/lalith/mtl_scene_understanding_and_segmentation/checkpoints/stl_s/stl_s/epoch_train/checkpoint_D153_epoch.pth')
    elif args.model == 'v0_ng':
        pretrained_model = torch.load('/media/mobarak/data/lalith/mtl_scene_understanding_and_segmentation/checkpoints/stl_s_ng/stl_s_ng/epoch_train/checkpoint_D168_epoch.pth')
    elif args.model == 'v1':
        pretrained_model = torch.load('/media/mobarak/data/lalith/mtl_scene_understanding_and_segmentation/checkpoints/stl_s_v1/stl_s_v1/epoch_train/checkpoint_D168_epoch.pth')
    elif args.model == 'v2_gc':
        pretrained_model = torch.load('/media/mobarak/data/lalith/mtl_scene_understanding_and_segmentation/checkpoints/stl_s_v2_gc/stl_s_v2_gc/epoch_train/checkpoint_D168_epoch.pth')
    else:
        print("Choose weights")
        sys.exit()                    

    pretrained_dict = pretrained_model['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}
    model_dict.update(pretrained_dict) 
    model.load_state_dict(model_dict)

    return model

def label_to_index(lbl):
    return torch.tensor(map_dict.index(lbl))


def index_to_label(index):
    return map_dict[index]


def batch_intersection_union(predict, target, nclass):
    _, predict = torch.max(predict, 1)

    minim = 1
    maxim = nclass
    nbins = nclass

    predict = predict.cpu().numpy().astype('int64') + 1
    target = target.cpu().numpy().astype('int64') + 1

    # print(predict)
    # print("-" * 50)
    # print(target)

    predict = predict * (target > 0).astype(predict.dtype)
    intersection = predict * (predict == target)

    area_inter, _ = np.histogram(
        intersection, bins=nbins, range=(minim, maxim))
    area_pred, _ = np.histogram(predict, bins=nbins, range=(minim, maxim))
    area_lab, _ = np.histogram(target, bins=nbins, range=(minim, maxim))
    area_union = area_pred + area_lab - area_inter
    assert (area_inter <= area_union).all(), \
        "Intersection area should be smaller than Union area"
    return area_inter, area_union


def eval_pass(model, test_loader, nclass=8):
    model.eval()
    total_inter, total_union = 0, 0
    class_values = np.zeros(nclass)

    for data in tqdm(test_loader):
        seg_img = data['img']
        seg_masks = data['mask']
        img_loc = data['img_loc']
        node_num = data['node_num']
        roi_labels = data['roi_labels']
        det_boxes = data['det_boxes']
        edge_labels = data['edge_labels']
        spatial_feat = data['spatial_feat']
        word2vec = data['word2vec']

        seg_img, seg_masks = seg_img.cuda(non_blocking=True), seg_masks.cuda(non_blocking=True)
        with torch.no_grad():
            _, seg_outputs = model(seg_img, img_loc, det_boxes, node_num, spatial_feat, word2vec, roi_labels, validation=True)

            inter, union = batch_intersection_union(
                seg_outputs.data, seg_masks, nclass)

        total_inter += inter
        total_union += union

    IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
    class_values += IoU
    # class_values /= len(test_loader)
    return class_values


if __name__ == '__main__':

    # /media/mobarak/data/lalith/mtl_scene_understanding_and_segmentation/checkpoints/stl_s_v1/stl_s_v1/epoch_train/checkpoint_D168_epoch.pth
    # /media/mobarak/data/lalith/mtl_scene_understanding_and_segmentation/checkpoints/stl_s_v2_gc/stl_s_v2_gc/epoch_train/checkpoint_D168_epoch.pth
    parser = argparse.ArgumentParser(description='MTL Scene graph and segmentation')
    args = parser.parse_args()
    args.use_cbs = False
    args.num_gpu = torch.cuda.device_count()
    args.nodes = 1
    args.nr = 0
    args.global_feat = 0
    args.model = 'v0' # v1
    args.feature_extractor = 'resnet18_11_cbs_ts'
    args.world_size = args.num_gpu * args.nodes
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    label_path = '/media/mobarak/data/lalith/sai/eval_models/labels_isi_dataset.json'

    with open(label_path) as f:
        labels = json.load(f)

    CLASSES = []
    CLASS_ID = []

    for item in labels:
        CLASSES.append(item['name'])
        CLASS_ID.append(item['classid'])

    map_dict = {k: v for k, v in zip(CLASS_ID, CLASSES)}
    print(map_dict)

    train_seq = [[2, 3, 4, 6, 7, 9, 10, 11, 12, 14, 15]]
    val_seq = [[1, 5, 16]]
    # val_seq = [[1]]
    data_dir = ['/media/mobarak/data/lalith/mtl_scene_understanding_and_segmentation/datasets/instruments18/seq_']
    img_dir = ['/left_frames/']
    mask_dir = ['/annotations/']
    dset = [0]
    data_const = SurgicalSceneConstants()

    seq = {'train_seq': train_seq, 'val_seq': val_seq, 'data_dir': data_dir, 
            'img_dir': img_dir, 'dset': dset, 'mask_dir': mask_dir}

    # val_dataset only set in 1 GPU
    val_dataset = SurgicalSceneDataset(seq_set=seq['val_seq'], dset=seq['dset'], data_dir=seq['data_dir'], \
                                       img_dir=seq['img_dir'], mask_dir=seq['mask_dir'], istrain=False, dataconst=data_const, \
                                       feature_extractor=args.feature_extractor, reduce_size=False)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    # train_dataset distributed to 2 GPU
    train_dataset = SurgicalSceneDataset(seq_set=seq['train_seq'], data_dir=seq['data_dir'],
                                         img_dir=seq['img_dir'], mask_dir=seq['mask_dir'], dset=seq['dset'], istrain=True, dataconst=data_const,
                                         feature_extractor=args.feature_extractor, reduce_size=False)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    model = build_model(args, load_pretrained=False)
    model = load(args, model).to(device)

    # Priority rank given to node 0, current pc, more node means multiple PC
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8900'
    rank = args.nr * args.num_gpu + args.num_gpu
    dist.init_process_group(backend='nccl', init_method='env://', world_size=1, rank=0)

    # rank, args.world_size, args.nr, args.gpus, gpu - 0 1 0 1 0
    # fix seeds and set cuda
    seed_everything()  # seed all random
    torch.cuda.set_device(0)

    model.cuda()
    model = DDP(model, device_ids=[0], find_unused_parameters=True)
    print("After DDP")

    class_values = eval_pass(model, val_dataloader)
    class_wise_IoU = []
    m_vals = []
    for idx, value in enumerate(class_values):
        class_name = index_to_label(idx)
        pair = [class_name, value]
        m_vals.append(value)
        class_wise_IoU.append(pair)

    print(class_wise_IoU)
    print("Mean Value: ", np.mean(np.array(m_vals)))

    print(tabulate(class_wise_IoU,
          headers=['Class', 'IoU'], tablefmt='orgtbl'))