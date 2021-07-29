import os
import copy
import time

import cv2
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image


import torch
import torchvision
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
#from tensorboardX import SummaryWriter


from models.feature_encoder import *
from models.scene_graph import *
from models.surgicalDataset import *

import utils.io as io
from utils.vis_tool import vis_img
from utils.scene_graph_eval_matrix import *


import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda import amp


def seed_everything(seed=27):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class mtl_model(nn.Module):
    '''
    Multi-task model : Graph Scene Understanding and segmentation
    Forward uses features from feature_extractor
    '''

    def __init__(self, feature_encoder, scene_graph):
        super(mtl_model, self).__init__()
        self.feature_encoder = feature_encoder
        self.scene_graph = scene_graph

        self.transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    def forward(self, img_dir, det_boxes_all, node_num, spatial_feat, word2vec, roi_labels, validation=False):               
        
        fe_feature = None
        gsu_node_feat = None
        
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
                if img_stack is None: img_stack = roi_image
                else: img_stack = torch.cat((img_stack, roi_image))
            
            img_stack = img_stack.cuda(non_blocking=True)
            img_stack = self.feature_encoder(img_stack)
            
            # prepare FE
            if fe_feature == None: fe_feature = img_stack
            else: fe_feature = torch.cat((fe_feature,img_stack))
            
            # prepare graph node features
            if gsu_node_feat == None: gsu_node_feat = img_stack.view(img_stack.size(0), -1)
            else: gsu_node_feat = torch.cat((gsu_node_feat, img_stack.view(img_stack.size(0), -1)))
        
        # graph su model
        interaction = self.scene_graph(node_num, gsu_node_feat, spatial_feat, word2vec, roi_labels, validation= validation)
        return interaction
        #return fe_feature, interaction


def build_model(args, pretrained_model = True):
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
    if pretrained_model:
        pretrained_model = torch.load(args.gsu_checkpoint)
        scene_graph.load_state_dict(pretrained_model['state_dict'])
    #scene_graph.eval()

    '''==== Feature extractor ===='''
    # feature extraction model
    feature_encoder = ResNet18(args)
    if args.use_cbs: feature_encoder.get_new_kernels(0)
    
    # based on cuda
    num_gpu = torch.cuda.device_count()
    if num_gpu > 0:
       device_ids = np.arange(num_gpu).tolist() 
       feature_encoder = nn.DataParallel(feature_encoder, device_ids=device_ids)
    
    # feature extraction pre-trained weights
    feature_encoder.load_state_dict(torch.load(args.fe_modelpath))
    feature_encoder = feature_encoder.module
    
    if args.use_cbs: feature_encoder = nn.Sequential(*list(feature_encoder.children())[:-2])
    else: feature_encoder = nn.Sequential(*list(feature_encoder.children())[:-1])

    model = mtl_model(feature_encoder, scene_graph)
    model.to(torch.device('cpu'))
    return model


def model_eval(model, validation_dataloader):
    '''
    Evaluate MTL
    '''
    
    model.eval()

    # graph
    scene_graph_criterion = nn.MultiLabelSoftMarginLoss()                   
    scene_graph_edge_count = 0
    scene_graph_total_acc = 0.0
    scene_graph_total_loss = 0.0
    scene_graph_logits_list = []
    scene_graph_labels_list = []
    
    for data in tqdm(validation_dataloader):
    #for data in train_dataloader:
            
        img_loc = data['img_loc']
        node_num = data['node_num']
        roi_labels = data['roi_labels']
        det_boxes = data['det_boxes']
        edge_labels = data['edge_labels']
        spatial_feat = data['spatial_feat']
        word2vec = data['word2vec']
        spatial_feat, word2vec, edge_labels = spatial_feat.cuda(non_blocking=True), word2vec.cuda(non_blocking=True), edge_labels.cuda(non_blocking=True)  

        with torch.no_grad():
            interaction = model(img_loc, det_boxes, node_num, spatial_feat, word2vec, roi_labels, validation=True)
                    
        scene_graph_logits_list.append(interaction)
        scene_graph_labels_list.append(edge_labels)
            
        # loss and accuracy
        scene_graph_loss = scene_graph_criterion(interaction, edge_labels.float())
        scene_graph_acc = np.sum(np.equal(np.argmax(interaction.cpu().data.numpy(), axis=-1), np.argmax(edge_labels.cpu().data.numpy(), axis=-1)))
            
        # accumulate loss and accuracy of the batch
        scene_graph_total_loss += scene_graph_loss.item() * edge_labels.shape[0]
        scene_graph_total_acc += scene_graph_acc
        scene_graph_edge_count += edge_labels.shape[0]
        
    # graph evaluation
    scene_graph_total_acc = scene_graph_total_acc / scene_graph_edge_count
    scene_graph_total_loss = scene_graph_total_loss / len(validation_dataloader)
    scene_graph_logits_all = torch.cat(scene_graph_logits_list).cuda()
    scene_graph_labels_all = torch.cat(scene_graph_labels_list).cuda()
    scene_graph_logits_all = F.softmax(scene_graph_logits_all, dim=1)
    scene_graph_map_value, scene_graph_recall = calibration_metrics(scene_graph_logits_all, scene_graph_labels_all)

    print('================= Evaluation ====================')
    print('Graph : {acc: %0.4f map: %0.4f recall: %0.4f}' % (scene_graph_total_acc, scene_graph_map_value, scene_graph_recall))
    

def train_model(gpu, args):
    '''
    training MTL model
    '''
    
    # this is placed above the dist.init process, possibility because of the feature_extraction model.
    model = build_model(args, pretrained_model=False)
    
    # Priority rank given to node 0, current pc, more node means multiple PC
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8892'
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    
    # fix seeds and set cuda
    seed_everything()  # seed all random
    torch.cuda.set_device(gpu)
    
    # Wrap the model with ddp
    model.cuda()    
    model = DDP(model, device_ids=[gpu])

    # define loss function (criterion) and optimizer 
    graph_scene_criterion = nn.MultiLabelSoftMarginLoss().cuda(gpu)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)

    # train and test dataloader
    train_seq = [[2, 3, 4, 6, 7, 9, 10, 11, 12, 14, 15]]
    val_seq = [[1, 5, 16]]
    data_dir = ['datasets/instruments18/seq_']
    img_dir = ['/left_frames/']
    dset = [0] 
    data_const = SurgicalSceneConstants()
    seq = {'train_seq': train_seq, 'val_seq': val_seq, 'data_dir': data_dir, 'img_dir':img_dir, 'dset': dset}
    # val_dataset only set in 1 GPU
    val_dataset = SurgicalSceneDataset(seq_set=seq['val_seq'], data_dir=seq['data_dir'], \
                        img_dir=seq['img_dir'], dset=seq['dset'], dataconst=data_const, \
                        feature_extractor=args.feature_extractor, reduce_size=False)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    
    # train_dataset distributed to 2 GPU
    train_dataset = SurgicalSceneDataset(seq_set = seq['train_seq'], data_dir = seq['data_dir'], \
                        img_dir = seq['img_dir'], dset = seq['dset'], dataconst = data_const, \
                        feature_extractor = args.feature_extractor, reduce_size = False)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=args.world_size, rank=rank, shuffle=True)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=True, sampler=train_sampler)
    
    for epoch_count in range(args.epoch):
        
        start_time = time.time()
        model.train()

        # each epoch has a training and validation step
        epoch_acc = 0
        epoch_loss = 0

        running_acc = 0.0
        running_loss = 0.0
        running_edge_count = 0
        #idx = 0
            
        # E-cbs
        if args.use_cbs:
            model.module.scene_graph.grnn1.gnn.apply_h_h_edge.get_new_kernels(epoch_count)
            model = model.cuda()

        train_sampler.set_epoch(epoch_count)

        for data in tqdm(train_dataloader):
            
            #model.zero_grad()
            img_loc = data['img_loc']
            node_num = data['node_num']
            roi_labels = data['roi_labels']
            det_boxes = data['det_boxes']
            edge_labels = data['edge_labels']
            spatial_feat = data['spatial_feat']
            word2vec = data['word2vec']

            spatial_feat, word2vec, edge_labels = spatial_feat.cuda(non_blocking=True), word2vec.cuda(non_blocking=True), edge_labels.cuda(non_blocking=True) 
            
            # forward_propagation
            interaction = model(img_loc, det_boxes, node_num, spatial_feat, word2vec, roi_labels)
            
            # loss and acc calculation
            # loss and accuracy
            #if args.use_t: interaction = interaction / args.t_scale
            loss = graph_scene_criterion(interaction, edge_labels.float())
            acc = np.sum(np.equal(np.argmax(interaction.cpu().data.numpy(), axis=-1), np.argmax(edge_labels.cpu().data.numpy(), axis=-1)))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #    scaled_loss.backward()
            # idx+=1
            
            # accumulate loss of each batch
            running_loss += loss.item() * edge_labels.shape[0]
            running_acc += acc
            running_edge_count += edge_labels.shape[0]

        # calculate the loss and accuracy of each epoch
        epoch_loss = running_loss / len(train_dataloader)
        epoch_acc = running_acc / running_edge_count
            
        if gpu == 0:
            end_time = time.time()
            print('================= Train ====================')
            print("Train Epoch: {}/{} Acc: {:0.4f} Loss: {:0.4f} Execution time: {:0.4f}".format(\
                    epoch_count + 1, args.epoch, epoch_acc, epoch_loss, (end_time-start_time)))
        
            # save model
            # if epoch_loss<0.0405 or epoch_count % args.save_every == (args.save_every - 1):
            checkpoint = { 
                'lr': args.lr,
                'b_s': args.batch_size,
                'bias': args.bias, 
                'bn': args.bn, 
                'dropout': args.drop_prob,
                'layers': args.layers,
                'multi_head': args.multi_attn,
                'diff_edge': args.diff_edge,
                'state_dict': model.module.state_dict()
            }

            save_name = "checkpoint_D1" + str(epoch_count+1) + '_epoch.pth'
            torch.save(checkpoint, os.path.join(args.save_dir, args.exp_ver, 'epoch_train', save_name))

            model_eval(model, val_dataloader)

    return

    
if __name__ == "__main__":
     
    ver = 'mtl_test'
    f_e = 'resnet18_09_cbs_ts'

    # os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"
    num_gpu = torch.cuda.device_count()

    # arguments
    parser = argparse.ArgumentParser(description='MTL Scene graph and segmentation')

    # hyper parameters
    parser.add_argument('--lr',                 type=float,     default = 0.000001)
    parser.add_argument('--epoch',              type=int,       default = 80)
    parser.add_argument('--start_epoch',        type=int,       default = 0)
    parser.add_argument('--batch_size',         type=int,       default = 4)
    parser.add_argument('--gpu',                type=bool,      default = True)
    parser.add_argument('--print_every',        type=int,       default = 10)
    parser.add_argument('--train_model',        type=str,       default = 'epoch')
    parser.add_argument('--exp_ver',            type=str,       default = ver)

    # file locations
    parser.add_argument('--log_dir',            type=str,       default = './log/' + ver)
    parser.add_argument('--save_dir',           type=str,       default = './checkpoints/' + ver)
    parser.add_argument('--output_img_dir',     type=str,       default = './results/' + ver)
    parser.add_argument('--save_every',         type=int,       default = 10)
    parser.add_argument('--pretrained',         type=str,       default = None)

    # optimizer
    parser.add_argument('--optim',              type=str,       default = 'adam') # choices=['sgd', 'adam']

    # network
    parser.add_argument('--layers',             type=int,       default = 1)
    parser.add_argument('--bn',                 type=bool,      default = False)
    parser.add_argument('--drop_prob',          type=float,     default = 0.3)
    parser.add_argument('--bias',               type=bool,      default = True)
    parser.add_argument('--multi_attn',         type=bool,      default = False)
    parser.add_argument('--diff_edge',          type=bool,      default = False)
        
    # feature_encoder_modelpath
    parser.add_argument('--fe_modelpath',       type=str,       default = 'feature_extractor/checkpoint/incremental/inc_ResNet18_cbs_ts_0_012345678.pkl')
        
    # data_processing
    parser.add_argument('--sampler',            type=int,       default = 0)
    parser.add_argument('--data_aug',           type=bool,      default = False)
    parser.add_argument('--feature_extractor',  type=str,       default = f_e)
        
    # CBS
    parser.add_argument('--use_cbs',            type=bool,      default = True)
        
    # temperature_scaling
    parser.add_argument('--use_t',              type=bool,      default = False)
    parser.add_argument('--t_scale',            type=float,     default = 1.5)

    # gpu distributor
    parser.add_argument('--nodes',              type=int,       default = 1,    metavar='N',    help='number of data loading workers (default: 4)')
    parser.add_argument('--gpus',               type=int,       default = num_gpu,                help='number of gpus per node')
    parser.add_argument('--nr',                 type=int,       default = 0,                      help='ranking within the nodes')

    args = parser.parse_args()

    # seed_everything()
    data_const = SurgicalSceneConstants()

    # GPU distributed
    args.world_size = args.gpus * args.nodes

    # train model in distributed set
    # trainfunction, no of gpus, arguments
    mp.spawn(train_model, nprocs=args.gpus, args=(args,)) 