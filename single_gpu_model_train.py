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
from tensorboardX import SummaryWriter


import utils.io as io
from utils.vis_tool import vis_img
from models.feature_encoder import *
from models.scene_graph import *
from models.surgicalDataset import *


import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


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
        self.transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        self.scene_graph = scene_graph
    
    def forward(self, img_dir, det_boxes_all, node_num, spatial_feat, word2vec, roi_labels, validation = False):               
        
        fe_feature = None
        gsu_node_feat = None
        
        # feature extraction model
        for index, img_loc in  enumerate(img_dir):
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
            
            img_stack = img_stack.cuda()
            img_stack = self.feature_encoder(img_stack)
            
            # prepare FE
            if fe_feature == None: fe_feature = img_stack
            else: fe_feature = torch.cat((fe_feature,img_stack))
            
            # prepare graph node features
            if gsu_node_feat == None: gsu_node_feat = img_stack.view(img_stack.size(0), -1)
            else: gsu_node_feat = torch.cat((gsu_node_feat,img_stack.view(img_stack.size(0), -1)))

        #fe_feature = features
        #gsu_node_feat = features
        
        # graph su model
        interaction = self.scene_graph(node_num, gsu_node_feat, spatial_feat, word2vec, roi_labels, validation= validation)
        
        return fe_feature, interaction


def build_model(args, device, pretrained_model = True):
    '''
    Build MTL model
    1) Feature Extraction
    3) Graph Scene Understanding Model
    '''

    '''==== graph model ===='''
    # graph model
    scene_graph = AGRNN(bias= True, bn= False, dropout=0.3, multi_attn=False, layer=1, diff_edge=False, use_cbs = args.use_cbs)
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
    model = model.to(device)
    return model


def run_model(args, data_const):
    '''
    
    '''
    os.environ["CUDA_VISIBLE_DEVICES"]="1,2"
    device = torch.device('cuda')
    
    print('training on {}...'.format(device))
 
    # build mtl
    model = build_model(args, device, pretrained_model = False)

    # load pretrained model
    if args.pretrained:
        print(f"loading pretrained model {args.pretrained}")
        checkpoints = torch.load(args.pretrained, map_location=device)
        model.load_state_dict(checkpoints['state_dict'])
    model = model.to(device)
    
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.3) #the scheduler divides the lr by 10 every 150 epochs

    # get the configuration of the model and save some key configurations
    io.mkdir_if_not_exists(os.path.join(args.save_dir, args.exp_ver), recursive=True)
    for i in range(args.layers):
        if i==0:
            model_config = model.scene_graph.CONFIG1.save_config()
            model_config['lr'] = args.lr
            model_config['bs'] = args.batch_size
            model_config['layers'] = args.layers
            model_config['multi_attn'] = args.multi_attn
            model_config['data_aug'] = args.data_aug
            model_config['drop_out'] = args.drop_prob
            model_config['optimizer'] = args.optim
            model_config['diff_edge'] = args.diff_edge
            #model_config['model_parameters'] = parameter_num
            io.dump_json_object(model_config, os.path.join(args.save_dir, args.exp_ver, 'l1_config.json'))
    print('save key configurations successfully...')

    # domain 1
    train_seq = [[2,3,4,6,7,9,10,11,12,14,15]]
    val_seq = [[1,5,16]]
    data_dir = ['datasets/instruments18/seq_']
    img_dir = ['/left_frames/']
    dset = [0] # 0 for ISC, 1 for SGH
    seq = {'train_seq': train_seq, 'val_seq': val_seq, 'data_dir': data_dir, 'img_dir':img_dir, 'dset': dset}
    
    print('======================== commencing training ==============================')
    epoch_train(args, model,seq, device, "D1")


def epoch_train(args, model, seq, device, dname, finetune = False):
    '''
    input: model, dataloader, dataset, criterain, optimizer, scheduler, device, data_const
    data: 
        img_name, node_num, roi_labels, det_boxes, edge_labels,
        edge_num, features, spatial_features, word2vec
    '''

    stop_epoch = args.epoch
    
    # set up dataset variable
    train_dataset = SurgicalSceneDataset(seq_set = seq['train_seq'], data_dir = seq['data_dir'], \
                        img_dir = seq['img_dir'], dset = seq['dset'], dataconst = data_const, \
                        feature_extractor = args.feature_extractor, reduce_size = False)
    val_dataset = SurgicalSceneDataset(seq_set = seq['val_seq'], data_dir = seq['data_dir'], \
                        img_dir = seq['img_dir'], dset = seq['dset'], dataconst = data_const, \
                        feature_extractor = args.feature_extractor, reduce_size = False)
    dataset = {'train': train_dataset, 'val': val_dataset}

    # use default DataLoader() to load the data. 
    train_dataloader = DataLoader(dataset=dataset['train'], batch_size=args.batch_size, shuffle= True, collate_fn=collate_fn)
    val_dataloader = DataLoader(dataset=dataset['val'], batch_size=args.batch_size, shuffle= True, collate_fn=collate_fn)
    dataloader = {'train': train_dataloader, 'val': val_dataloader}
    
    # criterion and scheduler
    criterion = nn.MultiLabelSoftMarginLoss()
    
    # set visualization and create folder to save checkpoints
    writer = SummaryWriter(log_dir=args.log_dir + '/' + args.exp_ver + '/' + 'epoch_train')
    io.mkdir_if_not_exists(os.path.join(args.save_dir, args.exp_ver, 'epoch_train'), recursive=True)

    for epoch in range(args.start_epoch, stop_epoch):
        
        # each epoch has a training and validation step
        epoch_acc = 0
        epoch_loss = 0
        
        #    optimizer = optim.SGD(model.parameters(), lr= args.lr, momentum=0.9, weight_decay=0)
        optimizer = optim.Adam(model.parameters(), lr= args.lr, weight_decay=0)
        
        for phase in ['train', 'val']:
            
            start_time = time.time()
            
            idx = 0
            running_acc = 0.0
            running_loss = 0.0
            running_edge_count = 0
            
            if phase == 'train' and args.use_cbs:
                model.scene_graph.grnn1.gnn.apply_h_h_edge.get_new_kernels(epoch)
                model = model.to(device)
            
            #print(len(dataloader[phase]))
            for data in tqdm(dataloader[phase]):
            #for data in dataloader[phase]:
                train_data = data
                img_name = train_data['img_name']
                img_loc = train_data['img_loc']
                node_num = train_data['node_num']
                roi_labels = train_data['roi_labels']
                det_boxes = train_data['det_boxes']
                edge_labels = train_data['edge_labels']
                edge_num = train_data['edge_num']
                # features = train_data['features']
                spatial_feat = train_data['spatial_feat']
                word2vec = train_data['word2vec']
                spatial_feat, word2vec, edge_labels = spatial_feat.to(device), word2vec.to(device), edge_labels.to(device)    
                
                #print(img_loc, det_boxes)
                if phase == 'train':
                    model.train()
                    model.zero_grad()
                    _, outputs = model(train_data['img_loc'], det_boxes, node_num, spatial_feat, word2vec, roi_labels)
                    
                    # loss and accuracy
                    if args.use_t: outputs = outputs / args.t_scale
                    loss = criterion(outputs, edge_labels.float())
                    loss.backward()
                    optimizer.step()
                    acc = np.sum(np.equal(np.argmax(outputs.cpu().data.numpy(), axis=-1), np.argmax(edge_labels.cpu().data.numpy(), axis=-1)))

                else:
                    model.eval()
                    # turn off the gradients for validation, save memory and computations
                    with torch.no_grad():
                        _, outputs = model(img_loc, det_boxes, node_num, spatial_feat, word2vec, roi_labels, validation=True)
                        
                        # loss and accuracy
                        loss = criterion(outputs, edge_labels.float())
                        acc = np.sum(np.equal(np.argmax(outputs.cpu().data.numpy(), axis=-1), np.argmax(edge_labels.cpu().data.numpy(), axis=-1)))
                    
                        # print result every 1000 iteration during validation
                        if idx == 10:
                            #print(img_loc[0])
                            io.mkdir_if_not_exists(os.path.join(args.output_img_dir, ('epoch_'+str(epoch))), recursive=True)
                            image = Image.open(img_loc[0]).convert('RGB')
                            det_actions = nn.Sigmoid()(outputs[0:int(edge_num[0])])
                            det_actions = det_actions.cpu().detach().numpy()
                            action_img = vis_img(image, roi_labels[0], det_boxes[0],  det_actions, data_const, score_thresh = 0.7)
                            image = image.save(os.path.join(args.output_img_dir, ('epoch_'+str(epoch)),img_name[0]))

                idx+=1
                # accumulate loss of each batch
                running_loss += loss.item() * edge_labels.shape[0]
                running_acc += acc
                running_edge_count += edge_labels.shape[0]
            
            # calculate the loss and accuracy of each epoch
            epoch_loss = running_loss / len(dataset[phase])
            epoch_acc = running_acc / running_edge_count
            
            # import ipdb; ipdb.set_trace()
            # log trainval datas, and visualize them in the same graph
            if phase == 'train': train_loss = epoch_loss 
            else: writer.add_scalars('trainval_loss_epoch', {'train': train_loss, 'val': epoch_loss}, epoch)
            
            # print data
            if (epoch % args.print_every) == 0:
                end_time = time.time()
                print("[{}] Epoch: {}/{} Acc: {:0.6f} Loss: {:0.6f} Execution time: {:0.6f}".format(\
                        phase, epoch+1, args.epoch, epoch_acc, epoch_loss, (end_time-start_time)))
                        
        # scheduler.step()
        # save model
        if epoch_loss<0.0405 or epoch % args.save_every == (args.save_every - 1) and epoch >= (20-1):
            checkpoint = { 
                            'lr': args.lr,
                           'b_s': args.batch_size,
                          'bias': args.bias, 
                            'bn': args.bn, 
                       'dropout': args.drop_prob,
                        'layers': args.layers,
                    'multi_head': args.multi_attn,
                     'diff_edge': args.diff_edge,
                    'state_dict': model.state_dict()
            }
            save_name = "checkpoint_" + dname + str(epoch+1) + '_epoch.pth'
            torch.save(checkpoint, os.path.join(args.save_dir, args.exp_ver, 'epoch_train', save_name))

    writer.close()

# setup process
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    dist.init_process_group("mtl", rank=rank, world_size=world_size)

# clean up
def cleanup():
    dist.destroy_process_group()

    
if __name__ == "__main__":
     
    ver = 'mtl_test'
    f_e = 'resnet18_09_cbs_ts'

    # arguments
    parser = argparse.ArgumentParser(description='MTL Scene graph and segmentation')

    # hyper parameters
    parser.add_argument('--lr',                 type=float,     default = 0.0000005)
    parser.add_argument('--epoch',              type=int,       default = 251)
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
    parser.add_argument('--optim',              type=str,       default='adam') # choices=['sgd', 'adam']

    # network
    parser.add_argument('--layers',             type=int,       default= 1)
    parser.add_argument('--bn',                 type=bool,       default = False)
    parser.add_argument('--drop_prob',          type=float,       default = 0.3)
    parser.add_argument('--bias',               type=bool,       default = True)
    parser.add_argument('--multi_attn',         type=bool,       default = False)
    parser.add_argument('--diff_edge',          type=bool,       default = False)
        
    # feature_encoder_modelpath
    parser.add_argument('--fe_modelpath',       type=str,       default = 'feature_extractor/checkpoint/incremental/inc_ResNet18_cbs_ts_0_012345678.pkl')
        
    # data_processing
    parser.add_argument('--sampler',            type=int,       default = 0)
    parser.add_argument('--data_aug',           type=bool,       default = False)
    parser.add_argument('--feature_extractor',  type=str,       default = f_e)
        
    # CBS
    parser.add_argument('--use_cbs',            type=bool,       default = True)
        
    # temperature_scaling
    parser.add_argument('--use_t',              type=bool,       default = True)
    parser.add_argument('--t_scale',            type=float,       default = 1.5)
    args = parser.parse_args()

    seed_everything()
    data_const = SurgicalSceneConstants()
    run_model(args, data_const)
