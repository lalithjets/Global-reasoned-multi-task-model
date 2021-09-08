'''
Project         : Global-Reasoned Multi-Task Surgical Scene Understanding
Lab             : MMLAB, National University of Singapore
contributors    : Lalithkumar Seenivasan, Sai Mitheran, Mobarakol Islam, Hongliang Ren
'''

import os
import time

import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.mtl_model import *
from models.scene_graph import *
from models.surgicalDataset import *
from models.segmentation_model import get_gcnet 

from utils.scene_graph_eval_matrix import *
from utils.segmentation_eval_matrix import *  


import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def seed_everything(seed=27):
    '''
    Set random seed for reproducible experiments
    Inputs: seed number 
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seg_eval_batch(seg_output, target):
    '''
    Calculate segmentation loss, pixel acc and IoU
    Inputs: predicted segmentation mask, GT segmentation mask 
    '''
    seg_criterion = SegmentationLosses(se_loss=False, aux=False, nclass=8, se_weight=0.2, aux_weight=0.2)
    loss = seg_criterion(seg_output, target)
    correct, labeled = batch_pix_accuracy(seg_output.data, target)
    inter, union = batch_intersection_union(seg_output.data, target, 8)  # 8 is num classes
    return correct, labeled, inter, union, loss

def get_checkpoint_loc(model_type, seg_mode = None):
    loc = None
    if model_type == 'amtl-t0' or model_type == 'amtl-t3':
        if seg_mode is None:
            loc = 'checkpoints/stl_s/stl_s/epoch_train/checkpoint_D153_epoch.pth'
        elif seg_mode == 'v1':
            loc = 'checkpoints/stl_s_v1/stl_s_v1/epoch_train/checkpoint_D168_epoch.pth'
        elif seg_mode == 'v2_gc':
            loc = 'checkpoints/stl_s_v2_gc/stl_s_v2_gc/epoch_train/checkpoint_D168_epoch.pth'
    elif model_type == 'amtl-t1':
        loc = 'checkpoints/stl_s/stl_s/epoch_train/checkpoint_D168_epoch.pth'
    elif model_type == 'amtl-t2':
        loc = 'checkpoints/stl_sg_wfe/stl_sg_wfe/epoch_train/checkpoint_D110_epoch.pth'
    return loc

def build_model(args):
    '''
    Build MTL model
    1) Scene Graph Understanding Model
    2) Segmentation Model : Encoder, Reasoning unit, Decoder

    Inputs: args
    '''

    '''==== Graph model ===='''
    # graph model
    scene_graph = AGRNN(bias=True, bn=False, dropout=0.3, multi_attn=False, layer=1, diff_edge=False, global_feat=args.global_feat)

    # segmentation model
    seg_model = get_gcnet(backbone='resnet18_model', pretrained=True)
    model = mtl_model(seg_model.pretrained, scene_graph, seg_model.gr_interaction, seg_model.gr_decoder, seg_mode = args.seg_mode)
    model.to(torch.device('cpu'))
    return model


def model_eval(args, model, validation_dataloader):
    '''
    Evaluate function for the MTL model (Segmentation and Scene Graph Performance)
    Inputs: args, model, val-dataloader

    '''

    model.eval()

    # graph
    scene_graph_criterion = nn.MultiLabelSoftMarginLoss()
    scene_graph_edge_count = 0
    scene_graph_total_acc = 0.0
    scene_graph_total_loss = 0.0
    scene_graph_logits_list = []
    scene_graph_labels_list = []

    test_seg_loss = 0.0
    total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
    
    for data in tqdm(validation_dataloader):
        seg_img = data['img']
        seg_masks = data['mask']
        img_loc = data['img_loc']
        node_num = data['node_num']
        roi_labels = data['roi_labels']
        det_boxes = data['det_boxes']
        edge_labels = data['edge_labels']
        spatial_feat = data['spatial_feat']
        word2vec = data['word2vec']

        spatial_feat, word2vec, edge_labels = spatial_feat.cuda(non_blocking=True), word2vec.cuda(non_blocking=True), edge_labels.cuda(non_blocking=True)
        seg_img, seg_masks = seg_img.cuda(non_blocking=True), seg_masks.cuda(non_blocking=True)

        with torch.no_grad():
            interaction, seg_outputs, _ = model(seg_img, img_loc, det_boxes, node_num, spatial_feat, word2vec, roi_labels, validation=True)

        scene_graph_logits_list.append(interaction)
        scene_graph_labels_list.append(edge_labels)

        # Loss and accuracy
        scene_graph_loss = scene_graph_criterion(interaction, edge_labels.float())
        scene_graph_acc = np.sum(np.equal(np.argmax(interaction.cpu().data.numpy(), axis=-1), np.argmax(edge_labels.cpu().data.numpy(), axis=-1)))
        correct, labeled, inter, union, t_loss = seg_eval_batch(seg_outputs, seg_masks)

        # Accumulate scene graph loss and acc
        scene_graph_total_loss += scene_graph_loss.item() * edge_labels.shape[0]
        scene_graph_total_acc += scene_graph_acc
        scene_graph_edge_count += edge_labels.shape[0]

        total_correct += correct
        total_label += labeled
        total_inter += inter
        total_union += union
        test_seg_loss += t_loss.item()

    # Graph evaluation
    scene_graph_total_acc = scene_graph_total_acc / scene_graph_edge_count
    scene_graph_total_loss = scene_graph_total_loss / len(validation_dataloader)
    scene_graph_logits_all = torch.cat(scene_graph_logits_list).cuda()
    scene_graph_labels_all = torch.cat(scene_graph_labels_list).cuda()
    scene_graph_logits_all = F.softmax(scene_graph_logits_all, dim=1)
    scene_graph_map_value, scene_graph_recall = calibration_metrics(scene_graph_logits_all, scene_graph_labels_all)

    # Segmentation evaluation
    pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
    IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
    mIoU = IoU.mean()

    print('================= Evaluation ====================')
    print('Graph        :  acc: %0.4f  map: %0.4f recall: %0.4f  loss: %0.4f}' % (scene_graph_total_acc, scene_graph_map_value, scene_graph_recall, scene_graph_total_loss))
    print('Segmentation : Pacc: %0.4f mIoU: %0.4f   loss: %0.4f}' % (pixAcc, mIoU, test_seg_loss/len(validation_dataloader)))
    return(scene_graph_total_acc, scene_graph_map_value, mIoU)


def train_model(gpu, args):
    '''
    Train function for the MTL model
    Inputs:  number of gpus per node, args

    '''
    # Store best value and epoch number
    best_value = [0.0, 0.0, 0.0]
    best_epoch = [0, 0, 0]

    # Decaying learning rate
    decay_lr = args.lr

    # This is placed above the dist.init process, because of the feature_extraction model.
    model = build_model(args)

    # Load pre-trained weights
    if args.model == 'amtl-t0' or args.model == 'amtl-t3' or args.model == 'amtl-t0-ft' or args.model == 'amtl-t1' or args.model == 'amtl-t2':
        print('Loading pre-trained weights for Sequential Optimisation')
        pretrained_model = torch.load(get_checkpoint_loc(args.model, args.seg_mode))
        pretrained_dict = pretrained_model['state_dict']
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}
        model_dict.update(pretrained_dict) 
        model.load_state_dict(model_dict)

    # Set training flag for submodules based on train model.
    model.set_train_test(args.model)


    if args.KD:
        teacher_model = build_model(args, load_pretrained=False)
        # Load pre-trained stl_mtl_model
        print('Preparing teacher model')
        pretrained_model = torch.load('/media/mobarak/data/lalith/mtl_scene_understanding_and_segmentation/checkpoints/stl_s_v1/stl_s_v1/epoch_train/checkpoint_D168_epoch.pth')
        pretrained_dict = pretrained_model['state_dict']
        model_dict = teacher_model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}
        model_dict.update(pretrained_dict) 
        teacher_model.load_state_dict(model_dict)
        if args.model == 'mtl-t3':
            teacher_model.set_train_test('mtl-t3')
            teacher_model.model_type3_insert()
            teacher_model.cuda()
        else:
            teacher_model.set_train_test('stl-s')
        teacher_model.cuda()
        teacher_model.eval()

    # Insert nn layers based on type.
    if args.model == 'amtl-t1' or args.model == 'mtl-t1':
        model.model_type1_insert()
    elif args.model == 'amtl-t2' or args.model == 'mtl-t2':
        model.model_type2_insert()
    elif args.model == 'amtl-t3' or args.model == 'mtl-t3':
        model.model_type3_insert()

    # Priority rank given to node 0 -> current pc, if more nodes -> multiple PCs
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port #8892
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)

    # Set cuda
    torch.cuda.set_device(gpu)

    # Wrap the model with ddp
    model.cuda()
    model = DDP(model, device_ids=[gpu], find_unused_parameters=True)#, find_unused_parameters=True)

    # Define loss function (criterion) and optimizer
    seg_criterion = SegmentationLosses(se_loss=False, aux=False, nclass=8, se_weight=0.2, aux_weight=0.2).cuda(gpu)
    graph_scene_criterion = nn.MultiLabelSoftMarginLoss().cuda(gpu)
    
    # train and test dataloader
    train_seq = [[2, 3, 4, 6, 7, 9, 10, 11, 12, 14, 15]]
    val_seq = [[1, 5, 16]]
    data_dir = ['datasets/instruments18/seq_']
    img_dir = ['/left_frames/']
    mask_dir = ['/annotations/']
    dset = [0]
    data_const = SurgicalSceneConstants()

    seq = {'train_seq': train_seq, 'val_seq': val_seq, 'data_dir': data_dir, 'img_dir': img_dir, 'dset': dset, 'mask_dir': mask_dir}

    # Val_dataset only set in 1 GPU
    val_dataset = SurgicalSceneDataset(seq_set=seq['val_seq'], dset=seq['dset'], data_dir=seq['data_dir'], \
                                       img_dir=seq['img_dir'], mask_dir=seq['mask_dir'], istrain=False, dataconst=data_const, \
                                       feature_extractor=args.feature_extractor, reduce_size=False)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # Train_dataset distributed to 2 GPU
    train_dataset = SurgicalSceneDataset(seq_set=seq['train_seq'], data_dir=seq['data_dir'],
                                         img_dir=seq['img_dir'], mask_dir=seq['mask_dir'], dset=seq['dset'], istrain=True, dataconst=data_const,
                                         feature_extractor=args.feature_extractor, reduce_size=False)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=args.world_size, rank=rank, shuffle=True)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=True, sampler=train_sampler)

    # Evaluate the model before start of training
    if gpu == 0:
        if args.KD:
            print("=================== Teacher Model=========================")
            eval_sc_acc, eval_sc_map, eval_seg_miou = model_eval(args, teacher_model, val_dataloader)
            print("=================== Student Model=========================")
        eval_sc_acc, eval_sc_map, eval_seg_miou = model_eval(args, model, val_dataloader)
        print("PT SC ACC: [value: {:0.4f}] PT SC mAP: [value: {:0.4f}] PT Seg mIoU: [value: {:0.4f}]".format(eval_sc_acc, eval_sc_map, eval_seg_miou))

    for epoch_count in range(args.epoch):

        start_time = time.time()

        # Set model / submodules in train mode
        model.train()
        if args.model == 'stl-sg' or args.model == 'amtl-t0' or args.model == 'amtl-t3':
            model.module.feature_encoder.eval()
            model.module.gcn_unit.eval()
            model.module.seg_decoder.eval()
        elif args.model == 'stl-sg-wfe':
            model.module.gcn_unit.eval()
            model.module.seg_decoder.eval()
        elif args.model == 'stl-s':
            model.module.scene_graph.eval()

        train_seg_loss = 0.0
        train_scene_graph_loss = 0.0

        model.cuda()

        # Optimizer with decaying learning rate
        decay_lr = decay_lr*0.98 if ((epoch_count+1) %10 == 0) else decay_lr
        optimizer = optim.Adam(model.parameters(), lr=decay_lr, weight_decay=0)

        train_sampler.set_epoch(epoch_count)

        if gpu == 0: print('================= Train ====================')

        for data in tqdm(train_dataloader):
            seg_img = data['img']
            seg_masks = data['mask']
            img_loc = data['img_loc']
            node_num = data['node_num']
            roi_labels = data['roi_labels']
            det_boxes = data['det_boxes']
            edge_labels = data['edge_labels']
            spatial_feat = data['spatial_feat']
            word2vec = data['word2vec']

            spatial_feat, word2vec, edge_labels = spatial_feat.cuda(non_blocking=True), word2vec.cuda(non_blocking=True), edge_labels.cuda(non_blocking=True)
            seg_img, seg_masks = seg_img.cuda(non_blocking=True), seg_masks.cuda(non_blocking=True)

            # Forward propagation
            interaction, seg_outputs, fe_feat = model(seg_img, img_loc, det_boxes, node_num, spatial_feat, word2vec, roi_labels)

            # Loss calculation
            seg_loss = seg_criterion(seg_outputs, seg_masks)
            scene_graph_loss = graph_scene_criterion(interaction, edge_labels.float())

            # KD-Loss
            if args.KD:
                with torch.no_grad():
                    _, _, t_fe_feat = teacher_model(seg_img, img_loc, det_boxes, node_num, spatial_feat, word2vec, roi_labels, validation=True)
                    t_fe_feat = t_fe_feat.detach()
                    t_fe_feat = t_fe_feat / (t_fe_feat.pow(2).sum(1) + 1e-6).sqrt().view(t_fe_feat.size(0), 1, t_fe_feat.size(2), t_fe_feat.size(3))
                
                    
                fe_feat = fe_feat
                fe_feat = fe_feat / (fe_feat.pow(2).sum(1) + 1e-6).sqrt().view(fe_feat.size(0), 1, fe_feat.size(2), fe_feat.size(3))
                dist_loss = (fe_feat - t_fe_feat).pow(2).sum(1).mean()
                        
                
            if args.model == 'stl-s':
                loss_total = seg_loss
            elif args.model == 'stl-sg' or args.model == 'stl-sg-wfe' or args.model == 'amtl-t0' or args.model == 'amtl-t3':
                loss_total = scene_graph_loss
            elif args.KD:
                loss_total = (0.4 * scene_graph_loss) + seg_loss + dist_loss
            else:
                loss_total = (0.4 * scene_graph_loss)+ (0.6 * seg_loss)
            
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            train_seg_loss += seg_loss.item()
            train_scene_graph_loss += scene_graph_loss.item() * edge_labels.shape[0]

        # calculate the loss and accuracy of each epoch
        train_seg_loss += train_seg_loss / len(train_dataloader)
        train_scene_graph_loss = train_scene_graph_loss / len(train_dataloader)

        if gpu == 0:
            end_time = time.time()
            print("Train Epoch: {}/{} lr: {:0.9f}  Graph_loss: {:0.4f} Segmentation_Loss: {:0.4f} Execution time: {:0.4f}".format(\
                    epoch_count + 1, args.epoch, decay_lr, train_scene_graph_loss, train_seg_loss, (end_time-start_time)))

            #if epoch_count % 2 == 0:
            # save model
            # if epoch_loss<0.0405 or epoch_count % args.save_every == (args.save_every - 1):
            checkpoint = {  'lr': args.lr, 'b_s': args.batch_size, 'bias': args.bias, 'bn': args.bn, 'dropout': args.drop_prob,
                            'layers': args.layers, 'multi_head': args.multi_attn,
                            'diff_edge': args.diff_edge, 'state_dict': model.module.state_dict() }

            save_name = "checkpoint_D1" + str(epoch_count+1) + '_epoch.pth'
            torch.save(checkpoint, os.path.join(args.save_dir, args.exp_ver, 'epoch_train', save_name))
            
            eval_sc_acc, eval_sc_map, eval_seg_miou = model_eval(args, model, val_dataloader)
            if eval_sc_acc > best_value[0]:
                best_value[0] = eval_sc_acc
                best_epoch[0] = epoch_count+1
            if eval_sc_map > best_value[1]:
                best_value[1] = eval_sc_map
                best_epoch[1] = epoch_count+1
            if eval_seg_miou > best_value[2]:
                best_value[2] = eval_seg_miou
                best_epoch[2] = epoch_count+1
            print("Best SC Acc: [Epoch: {} value: {:0.4f}] Best SC mAP: [Epoch: {} value: {:0.4f}] Best Seg mIoU: [Epoch: {} value: {:0.4f}]".format(\
                    best_epoch[0], best_value[0], best_epoch[1], best_value[1], best_epoch[2], best_value[2]))

    return


if __name__ == "__main__":
    '''
    Main function to set arguments
    '''

    # ---------------------------------------------- Optimization and feature sharing variants ----------------------------------------------
    '''
    Format for the model_type : X-Y 

    -> X : Optimisation technique [1. amtl - Sequential MTL Optimisation, 2. mtl - Naive MTL Optimisation]
    -> Y : Feature Sharing mechanism [1. t0 - Base model,
                                      2. t1 - Scene graph features to enhance segmentation (SGFSEG), 
                                      3. t3 - Global interaction space features to improve scene graph (GISFSG)]

    '''
    model_type = 'amtl-t0'
    ver = model_type + '_v5'
    port = '8892'
    f_e = 'resnet18_11_cbs_ts'


    #  ----------------------------------------------Global reasoning variant in segmentation -----------------------------------------------
    '''
    -> seg_mode : v1 - (MSLRGR - multi-scale local reasoning and global reasoning) 
                v2gc - (MSLR - multi-scale local reasoning) 
                None - Base model
    '''
    seg_mode = 'v1'
    
    # Set random seed
    seed_everything()  
    print(ver, seg_mode)

    # Device Count
    num_gpu = torch.cuda.device_count()

    # Arguments
    parser = argparse.ArgumentParser(description='MTL Scene graph and Segmentation')

    # Hyperparameters
    parser.add_argument('--lr',                 type=float,     default = 0.00001) 
    parser.add_argument('--epoch',              type=int,       default = 130)
    parser.add_argument('--start_epoch',        type=int,       default = 0)
    parser.add_argument('--batch_size',         type=int,       default = 4)
    parser.add_argument('--gpu',                type=bool,      default = True)
    parser.add_argument('--train_model',        type=str,       default = 'epoch')
    parser.add_argument('--exp_ver',            type=str,       default = ver)

    # File locations
    parser.add_argument('--log_dir',            type=str,       default = './log/' + ver)
    parser.add_argument('--save_dir',           type=str,       default = './checkpoints/' + ver)
    parser.add_argument('--output_img_dir',     type=str,       default = './results/' + ver)
    parser.add_argument('--save_every',         type=int,       default = 10)
    parser.add_argument('--pretrained',         type=str,       default = None)

    # Network settings
    parser.add_argument('--layers',             type=int,       default = 1)
    parser.add_argument('--bn',                 type=bool,      default = False)
    parser.add_argument('--drop_prob',          type=float,     default = 0.3)
    parser.add_argument('--bias',               type=bool,      default = True)
    parser.add_argument('--multi_attn',         type=bool,      default = False)
    parser.add_argument('--diff_edge',          type=bool,      default = False)

    if model_type == 'mtl-t3' or model_type == 'amtl-t3':
        parser.add_argument('--global_feat',        type=int,       default = 128)
    else:
        parser.add_argument('--global_feat',        type=int,       default = 0)

    # Data processing
    parser.add_argument('--sampler',            type=int,       default = 0)
    parser.add_argument('--data_aug',           type=bool,      default = False)
    parser.add_argument('--feature_extractor',  type=str,       default = f_e)
    parser.add_argument('--seg_mode',           type=str,       default = seg_mode) # v1/v2_gc
    
    parser.add_argument('--KD',                 type=bool,      default = False)

    # GPU distributor
    parser.add_argument('--port',               type=str,       default = port)
    parser.add_argument('--nodes',              type=int,       default = 1,        metavar='N',    help='Number of data loading workers (default: 4)')
    parser.add_argument('--gpus',               type=int,       default = num_gpu,                  help='Number of gpus per node')
    parser.add_argument('--nr',                 type=int,       default = 0,                        help='Ranking within the nodes')

    # Model type
    parser.add_argument('--model',              type=str,       default = model_type) 
    args = parser.parse_args()

    # Constants for the surgical scene
    data_const = SurgicalSceneConstants()

    # GPU distributed
    args.world_size = args.gpus * args.nodes

    # Train model in distributed settings - (train function, number of GPUs, arguments)
    mp.spawn(train_model, nprocs=args.gpus, args=(args,))