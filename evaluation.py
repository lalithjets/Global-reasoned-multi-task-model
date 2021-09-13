#from functools import lru_cache
import os
import time
import json

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
from models.segmentation_model import get_gcnet  # for the get_gcnet function

from utils.scene_graph_eval_matrix import *
from utils.segmentation_eval_matrix import *  # SegmentationLoss and Eval code

from tabulate import tabulate

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import warnings
warnings.filterwarnings('ignore')

def label_to_index(lbl):
    '''
    Label to index mapping
    Input: class label
    Output: class index
    '''
    return torch.tensor(map_dict.index(lbl))


def index_to_label(index):
    '''
    Index to label mapping
    Input: class index
    Output: class label
    '''
    return map_dict[index]



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
    '''
    seg_criterion = SegmentationLosses(se_loss=False, aux=False, nclass=8, se_weight=0.2, aux_weight=0.2)
    loss = seg_criterion(seg_output, target)
    correct, labeled = batch_pix_accuracy(seg_output.data, target)
    inter, union = batch_intersection_union(seg_output.data, target, 8)  # 8 is num classes
    return correct, labeled, inter, union, loss


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
    seg_model = get_gcnet(backbone='resnet18_model', pretrained=False)
    model = mtl_model(seg_model.pretrained, scene_graph, seg_model.gr_interaction, seg_model.gr_decoder, seg_mode = args.seg_mode)
    model.to(torch.device('cpu'))
    return model



def model_eval(model, validation_dataloader, nclass=8):
    '''
    Evaluate MTL
    '''

    model.eval()

    class_values = np.zeros(nclass)

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
    class_values += IoU
    mIoU = IoU.mean()

    print('\n================= Evaluation ====================')
    print('Graph        :  acc: %0.4f  map: %0.4f recall: %0.4f  loss: %0.4f}' % (scene_graph_total_acc, scene_graph_map_value, scene_graph_recall, scene_graph_total_loss))
    print('Segmentation : Pacc: %0.4f mIoU: %0.4f   loss: %0.4f}' % (pixAcc, mIoU, test_seg_loss/len(validation_dataloader)))

    print('\n================= Class-wise IoU ====================')
    class_wise_IoU = []
    m_vals = []
    for idx, value in enumerate(class_values):
        class_name = index_to_label(idx)
        pair = [class_name, value]
        m_vals.append(value)
        class_wise_IoU.append(pair)

    print("Mean Value: ", np.mean(np.array(m_vals)), "\n")

    print(tabulate(class_wise_IoU,
          headers=['Class', 'IoU'], tablefmt='orgtbl'))

    return(scene_graph_total_acc, scene_graph_map_value, mIoU)


if __name__ == "__main__":

    '''
    Main function to set arguments
    '''

    '''
    To reproduce the results, set the model_type, ver, seg_mode and checkpoint_dir based on the table below
    TBR = To be released
    ============================================================================================================
    Paper_name              |   model_type     |        ver           |    seg_mode    |   checkpoint_dir
    ============================================================================================================
                                                        STL
    ------------------------|-----------------------------------------------------------------------------------
    VS-GAT                  |    'stl-sg'      |    'stl_sg'          |      None      |    'stl_sg'
    SEG                     |    'stl-s'       |    'stl_s_ng'        |      TBR       |    'stl_s_ng'
    SEG-GR                  |    'stl-s'       |    'stl_s'           |      None      |    'stl_s'
    SEG-MSGR                |    'stl-s'       |    'stl_s_v2gc'      |     'v2gc'     |    'stl_s_v2gc'
    SEG-MSLRGR              |    'stl-s'       |    'stl_s_v1'        |     'v1'       |    'stl_s_v1'
    ------------------------------------------------------------------------------------------------------------
                                                       SMTL
    ------------------------------------------------------------------------------------------------------------
    GR                      |    'amtl-t0'     |    'amtl_t0_s'       |      None      |    'amtl_t0_s'
    MSGR                    |    'amtl-t0'     |    'amtl_t0_sv2gc'   |      'v2gc'    |    'amtl_t0_sv2gc'
    MSLRGR                  |    'amtl-t0'     |    'amtl_t0_sv1'     |      'v1'      |    'amtl_t0_sv1'
    MSLRGR-GISFSG           |    'amtl-t3'     |    'amtl_t3pn_sv1'   |      'v1'      |    'amtl_t3pn_sv1'          
    ------------------------------------------------------------------------------------------------------------
                                                      v-MTL
    ------------------------------------------------------------------------------------------------------------
    V-MTL-GR                |    'mtl-t0'      |    'mtl_t0_s         |      None      |    'mtl_t0_s'
    ------------------------------------------------------------------------------------------------------------
                                                        KD-MTL (set args.KD = True)
    ------------------------------------------------------------------------------------------------------------
    KD-MTL-GR               |    'mtl-t0'      |    'mtl_kd_t0_s'     |      None      |        TBR
    KD-MTL-MSLRGR           |    'mtl-t0'      |    'mtl_kd_t0_sv1'   |      'v1'      |    'mtl_kd_t0_sv1'
    KD-MTL-MSLRGR-SGFSEG    |    'mtl-t1'      |    'mtl_kd_t1_sv1'   |      'v1'      |    'mtl_kd_t1_sv1'
    KD-MTL-MSLRGR-GISFSG    |    'mtl-t3'      |    'mtl_kd_t3_sv1'   |      'v1'      |    'mtl_kd_t3_sv1'
    ------------------------------------------------------------------------------------------------------------
    '''

    model_type = 'amtl-t3'
    ver = 'amtl_t3_sv1'
    seg_mode = 'v1'
    checkpoint_dir = 'amtl_t3_sv1'

    port = '8892'

    # Set random seed
    seed_everything()  
    print(ver, seg_mode)

    # arguments
    parser = argparse.ArgumentParser(description='GR_MTL_SSU')

    # hyper parameters
    parser.add_argument('--lr',                 type=float,     default = 0.00001) #0.00001
    parser.add_argument('--epoch',              type=int,       default = 130)
    parser.add_argument('--start_epoch',        type=int,       default = 0)
    parser.add_argument('--batch_size',         type=int,       default = 1)
    parser.add_argument('--gpu',                type=bool,      default = True)
    parser.add_argument('--train_model',        type=str,       default = 'epoch')
    parser.add_argument('--exp_ver',            type=str,       default = ver)

    # file locations
    parser.add_argument('--log_dir',            type=str,       default = './log/' + ver)
    parser.add_argument('--save_dir',           type=str,       default = './checkpoints/' + ver)
    parser.add_argument('--output_img_dir',     type=str,       default = './results/' + ver)
    parser.add_argument('--save_every',         type=int,       default = 10)
    parser.add_argument('--pretrained',         type=str,       default = None)

    # network
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
    # data_processing
    parser.add_argument('--sampler',            type=int,       default = 0)
    parser.add_argument('--data_aug',           type=bool,      default = False)
    parser.add_argument('--feature_extractor',  type=str,       default = 'features')
    parser.add_argument('--seg_mode',           type=str,       default = seg_mode)

    # CBS
    parser.add_argument('--use_cbs',            type=bool,      default = False)

    # Knowledge distillation
    parser.add_argument('--KD',                 type=bool,      default = False)

    parser.add_argument('--model',              type=str,       default = model_type) 
    args = parser.parse_args()

    # seed_everything()
    data_const = SurgicalSceneConstants()

    label_path = 'dataset/labels_isi_dataset.json'
    with open(label_path) as f:
        labels = json.load(f)

    CLASSES = []
    CLASS_ID = []

    for item in labels:
        CLASSES.append(item['name'])
        CLASS_ID.append(item['classid'])

    map_dict = {k: v for k, v in zip(CLASS_ID, CLASSES)}

    # this is placed above the dist.init process, possibility because of the feature_extraction model.
    model = build_model(args)
    model.set_train_test(args.model)

    # insert nn layers based on type.
    if args.model == 'amtl-t1' or args.model == 'mtl-t1':
        model.model_type1_insert()
    elif args.model == 'amtl-t2' or args.model == 'mtl-t2':
        model.model_type2_insert()
    elif args.model == 'amtl-t3' or args.model == 'mtl-t3':
        model.model_type3_insert()

    # load pre-trained stl_mtl_model
    print('Loading pre-trained weights')
    pretrained_model = torch.load(('checkpoints/'+checkpoint_dir+'/best_epoch.pth'))
    model.load_state_dict(pretrained_model)
    
    # Wrap the model with ddp
    model.cuda()

    # train and test dataloader
    val_seq = [[1, 5, 16]]
    data_dir = ['dataset/seq_']
    img_dir = ['/left_frames/']
    mask_dir = ['/annotations/']
    dset = [0]
    data_const = SurgicalSceneConstants()

    seq = {'val_seq': val_seq, 'data_dir': data_dir, 'img_dir': img_dir, 'dset': dset, 'mask_dir': mask_dir}

    # val_dataset only set in 1 GPU
    val_dataset = SurgicalSceneDataset(seq_set=seq['val_seq'], dset=seq['dset'], data_dir=seq['data_dir'], \
                                       img_dir=seq['img_dir'], mask_dir=seq['mask_dir'], istrain=False, dataconst=data_const, \
                                       feature_extractor=args.feature_extractor, reduce_size=False)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    model_eval(model, val_dataloader)
