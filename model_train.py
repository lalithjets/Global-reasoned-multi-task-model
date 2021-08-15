from functools import lru_cache
import os
import copy
import time

import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image


import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
#from tensorboardX import SummaryWriter


# from models.feature_encoder import *
from models.mtl_model import *
from models.scene_graph import *
from models.surgicalDataset import *
from models.segmentation_model import get_gcnet  # for the get_gcnet function

import utils.io as io
from utils.vis_tool import vis_img
from utils.scene_graph_eval_matrix import *
from utils.segmentation_eval_matrix import *  # SegmentationLoss and Eval code


import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP



def seed_everything(seed=27):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seg_eval_batch(seg_output, target):
    '''
    Calculate segmentation loss, pixel acc and IoU
    '''
    seg_criterion = SegmentationLosses(se_loss=False, aux=False, nclass=8, se_weight=0.2, aux_weight=0.2)
    loss = seg_criterion(seg_output, target)
    correct, labeled = batch_pix_accuracy(seg_output.data, target)
    inter, union = batch_intersection_union(seg_output.data, target, 8)  # 8 is num classes
    return correct, labeled, inter, union, loss




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

    model = mtl_model(seg_model.pretrained, scene_graph, seg_model.gcn_block, seg_model.decoder)
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

    test_seg_loss = 0.0
    total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
    

    #tbar_val = tqdm(validation_dataloader)
    #for i, data in enumerate(tbar_val):
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
            interaction, seg_outputs = model(seg_img, img_loc, det_boxes, node_num, spatial_feat, word2vec, roi_labels, validation=True)

        scene_graph_logits_list.append(interaction)
        scene_graph_labels_list.append(edge_labels)

        # loss and accuracy
        scene_graph_loss = scene_graph_criterion(interaction, edge_labels.float())
        scene_graph_acc = np.sum(np.equal(np.argmax(interaction.cpu().data.numpy(), axis=-1), np.argmax(edge_labels.cpu().data.numpy(), axis=-1)))
        correct, labeled, inter, union, t_loss = seg_eval_batch(seg_outputs, seg_masks)

        # accumulate scene graph loss and acc
        scene_graph_total_loss += scene_graph_loss.item() * edge_labels.shape[0]
        scene_graph_total_acc += scene_graph_acc
        scene_graph_edge_count += edge_labels.shape[0]

        total_correct += correct
        total_label += labeled
        total_inter += inter
        total_union += union
        test_seg_loss += t_loss.item()
        # tbar_val.set_description('pixAcc: %.3f, mIoU: %.3f, Val-loss: %.3f' % (pixAcc, mIoU, test_seg_loss/(i + 1)))

    # graph evaluation
    scene_graph_total_acc = scene_graph_total_acc / scene_graph_edge_count
    scene_graph_total_loss = scene_graph_total_loss / len(validation_dataloader)
    scene_graph_logits_all = torch.cat(scene_graph_logits_list).cuda()
    scene_graph_labels_all = torch.cat(scene_graph_labels_list).cuda()
    scene_graph_logits_all = F.softmax(scene_graph_logits_all, dim=1)
    scene_graph_map_value, scene_graph_recall = calibration_metrics(scene_graph_logits_all, scene_graph_labels_all)

    # segmentation evaluation
    pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
    IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
    mIoU = IoU.mean()

    print('================= Evaluation ====================')
    print('Graph        :  acc: %0.4f  map: %0.4f recall: %0.4f  loss: %0.4f}' % (scene_graph_total_acc, scene_graph_map_value, scene_graph_recall, scene_graph_total_loss))
    print('Segmentation : Pacc: %0.4f mIoU: %0.4f   loss: %0.4f}' % (pixAcc, mIoU, test_seg_loss/len(validation_dataloader)))
    return(scene_graph_total_acc, scene_graph_map_value, mIoU)


def train_model(gpu, args):
    best_value = [0.0, 0.0, 0.0]
    best_epoch = [0, 0, 0]

    # for decaying lr
    decay_lr = args.lr

    '''
    training MTL model
    '''

    # this is placed above the dist.init process, possibility because of the feature_extraction model.
    model = build_model(args, load_pretrained=False)

    # Priority rank given to node 0, current pc, more node means multiple PC
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8892' #8892
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)

    # fix seeds and set cuda
    seed_everything()  # seed all random
    torch.cuda.set_device(gpu)

    # Wrap the model with ddp
    model.cuda()
    model = DDP(model, device_ids=[gpu], find_unused_parameters=True)#, find_unused_parameters=True)

    # define loss function (criterion) and optimizer
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

    # val_dataset only set in 1 GPU
    val_dataset = SurgicalSceneDataset(seq_set=seq['val_seq'], dset=seq['dset'], data_dir=seq['data_dir'], \
                                       img_dir=seq['img_dir'], mask_dir=seq['mask_dir'], istrain=False, dataconst=data_const, \
                                       feature_extractor=args.feature_extractor, reduce_size=False)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # train_dataset distributed to 2 GPU
    train_dataset = SurgicalSceneDataset(seq_set=seq['train_seq'], data_dir=seq['data_dir'],
                                         img_dir=seq['img_dir'], mask_dir=seq['mask_dir'], dset=seq['dset'], istrain=True, dataconst=data_const,
                                         feature_extractor=args.feature_extractor, reduce_size=False)
    print(len(train_dataset))
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=args.world_size, rank=rank, shuffle=True)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=True, sampler=train_sampler)

    for epoch_count in range(args.epoch):

        start_time = time.time()
        model.train()

        train_seg_loss = 0.0
        train_scene_graph_loss = 0.0

        if args.use_cbs and epoch_count<30: 
            model.module.feature_encoder.get_new_kernels(epoch_count)
            model.module.scene_graph.grnn1.gnn.apply_h_h_edge.get_new_kernels(epoch_count)
            model.cuda()

        # optimizer with decaying lr
        #decay_lr = decay_lr*0.98 if ((epoch_count+1) %20 == 0) else decay_lr        
        decay_lr = decay_lr*0.98 if ((epoch_count+1) %10 == 0) else decay_lr        
        optimizer = optim.Adam(model.parameters(), lr=decay_lr, weight_decay=0)

        train_sampler.set_epoch(epoch_count)

        if gpu == 0: print('================= Train ====================')
        #tbar = tqdm(train_dataloader)
        #for i, data in enumerate(tbar):
        for data in tqdm(train_dataloader):
            # model.zero_grad()
            seg_img = data['img']
            seg_masks = data['mask']
            #img_name = data['img_name']
            img_loc = data['img_loc']
            node_num = data['node_num']
            roi_labels = data['roi_labels']
            det_boxes = data['det_boxes']
            edge_labels = data['edge_labels']
            spatial_feat = data['spatial_feat']
            word2vec = data['word2vec']

            spatial_feat, word2vec, edge_labels = spatial_feat.cuda(non_blocking=True), word2vec.cuda(non_blocking=True), edge_labels.cuda(non_blocking=True)
            seg_img, seg_masks = seg_img.cuda(non_blocking=True), seg_masks.cuda(non_blocking=True)

            # forward_propagation
            interaction, seg_outputs = model(seg_img, img_loc, det_boxes, node_num, spatial_feat, word2vec, roi_labels)

            # loss calculation
            seg_loss = seg_criterion(seg_outputs, seg_masks)
            scene_graph_loss = graph_scene_criterion(interaction, edge_labels.float())
            #acc = np.sum(np.equal(np.argmax(interaction.cpu().data.numpy(), axis=-1), np.argmax(edge_labels.cpu().data.numpy(), axis=-1)))

            loss_total = (0.6 * seg_loss) + (0.4 * scene_graph_loss)  # ADDING BOTH THE LOSSES IN A NAIVE WAY
            #loss_total = seg_loss + scene_graph_loss  # ADDING BOTH THE LOSSES IN A NAIVE WAY
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            train_seg_loss += seg_loss.item()
            train_scene_graph_loss += scene_graph_loss.item() * edge_labels.shape[0]

            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #    scaled_loss.backward()
            # idx+=1

            #tbar.set_description('Train loss - SEG: %.3f' % (train_seg_loss / (i + 1)))
            # accumulate loss of each batch
            #running_acc += acc
            #running_edge_count += edge_labels.shape[0]

        # calculate the loss and accuracy of each epoch
        train_seg_loss += train_seg_loss / len(train_dataloader)
        train_scene_graph_loss = train_scene_graph_loss / len(train_dataloader)
        #epoch_acc = running_acc / running_edge_count

        if gpu == 0:
            end_time = time.time()
            print("Train Epoch: {}/{} lr: {:0.9f}  Graph_loss: {:0.4f} Segmentation_Loss: {:0.4f} Execution time: {:0.4f}".format(\
                    epoch_count + 1, args.epoch, decay_lr, train_scene_graph_loss, train_seg_loss, (end_time-start_time)))

            #if epoch_count % 2 == 0:
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
            
            eval_sc_acc, eval_sc_map, eval_seg_miou = model_eval(model, val_dataloader)
            if eval_sc_acc > best_value[0]:
                best_value[0] = eval_sc_acc
                best_epoch[0] = epoch_count+1
            if eval_sc_map > best_value[1]:
                best_value[1] = eval_sc_map
                best_epoch[1] = epoch_count+1
            if eval_seg_miou > best_value[2]:
                best_value[2] = eval_seg_miou
                best_epoch[2] = epoch_count+1
            print("Best SC ACC: [Epoch: {} value: {:0.4f}] Best SC mAP: [Epoch: {} value: {:0.4f}] Best Seg mIuU: [Epoch: {} value: {:0.4f}]".format(\
                    best_epoch[0], best_value[0], best_epoch[1], best_value[1], best_epoch[2], best_value[2]))

    return


if __name__ == "__main__":

    ver = 'mtl_base_eCBS'
    f_e = 'resnet18_11_cbs_ts'      # f_e = 'resnet18_09_cbs_ts' # CHECK

    # os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"
    num_gpu = torch.cuda.device_count()

    # arguments
    parser = argparse.ArgumentParser(description='MTL Scene graph and segmentation')

    # hyper parameters
    parser.add_argument('--lr',                 type=float,     default = 0.00001)#0.0000085)
    parser.add_argument('--epoch',              type=int,       default = 51)
    parser.add_argument('--start_epoch',        type=int,       default = 0)
    parser.add_argument('--batch_size',         type=int,       default = 4)
    parser.add_argument('--gpu',                type=bool,      default = True)
    #parser.add_argument('--print_every',        type=int,       default = 10)
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
    #parser.add_argument('--fe_modelpath',       type=str,       default = 'feature_extractor/checkpoint/incremental/inc_ResNet18_cbs_ts_0_012345678.pkl')
    #parser.add_argument('--fe_imgnet_path',       type=str,     default = 'models/r18/resnet18-f37072fd.pth')

    # data_processing
    parser.add_argument('--sampler',            type=int,       default = 0)
    parser.add_argument('--data_aug',           type=bool,      default = False)
    parser.add_argument('--feature_extractor',  type=str,       default = f_e)

    # CBS
    parser.add_argument('--use_cbs',            type=bool,      default = True)

    # gpu distributor
    parser.add_argument('--nodes',              type=int,       default = 1,        metavar='N',    help='number of data loading workers (default: 4)')
    parser.add_argument('--gpus',               type=int,       default = num_gpu,                  help='number of gpus per node')
    parser.add_argument('--nr',                 type=int,       default = 0,                        help='ranking within the nodes')

    args = parser.parse_args()

    # seed_everything()
    data_const = SurgicalSceneConstants()

    # GPU distributed
    args.world_size = args.gpus * args.nodes

    # train model in distributed set
    # trainfunction, no of gpus, arguments
    mp.spawn(train_model, nprocs=args.gpus, args=(args,))
