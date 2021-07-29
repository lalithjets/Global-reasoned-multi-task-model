import dgl
import math
import numpy as np

import torch
import torch
import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict


'''
configurations of the network
    
    readout: G_ER_L_S = [1024+300+16+300+1024,  1024, 117]

    node_func: G_N_L_S = [1024+1024, 1024]
    node_lang_func: G_N_L_S2 = [300+300+300]
    
    edge_func : G_E_L_S = [1024*2+16, 1024]
    edge_lang_func: [300*2, 1024]
    
    attn: [1024, 1]
    attn_lang: [1024, 1]
'''
class CONFIGURATION(object):
    '''
    Configuration arguments: feature type, layer, bias, batch normalization, dropout, multi-attn
    
    readout           : fc_size, activation, bias, bn, droupout
    gnn_node          : fc_size, activation, bias, bn, droupout
    gnn_node_for_lang : fc_size, activation, bias, bn, droupout
    gnn_edge          : fc_size, activation, bias, bn, droupout
    gnn_edge_for_lang : fc_size, activation, bias, bn, droupout
    gnn_attn          : fc_size, activation, bias, bn, droupout
    gnn_attn_for_lang : fc_size, activation, bias, bn, droupout
    '''
    def __init__(self, layer=1, bias=True, bn=False, dropout=0.2, multi_attn=False):
        
        # if multi_attn:
        if True:
            if layer==1:
                feature_size = 512
                # readout
                self.G_ER_L_S = [feature_size+300+16+300+feature_size, feature_size, 13]
                self.G_ER_A   = ['ReLU', 'Identity']
                self.G_ER_B   = bias    #true
                self.G_ER_BN  = bn      #false
                self.G_ER_D   = dropout #0.3
                # self.G_ER_GRU = feature_size

                # # gnn node function
                self.G_N_L_S = [feature_size+feature_size, feature_size]
                self.G_N_A   = ['ReLU']
                self.G_N_B   = bias #true
                self.G_N_BN  = bn      #false
                self.G_N_D   = dropout #0.3
                # self.G_N_GRU = feature_size

                # # gnn node function for language
                self.G_N_L_S2 = [300+300, 300]
                self.G_N_A2   = ['ReLU']
                self.G_N_B2   = bias    #true
                self.G_N_BN2  = bn      #false
                self.G_N_D2   = dropout #0.3
                # self.G_N_GRU2 = feature_size

                # gnn edge function1
                self.G_E_L_S           = [feature_size*2+16, feature_size]
                self.G_E_A             = ['ReLU']
                self.G_E_B             = bias     # true
                self.G_E_BN            = bn       # false
                self.G_E_D             = dropout  # 0.3
                self.G_E_c_std         = 1.0
                self.G_E_c_std_factor  = 0.985      # 0.985 (LOG), 0.95 (gau)
                self.G_E_c_epoch       = 20         # 20
                self.G_E_c_kernel_size = 3
                self.G_E_c_filter      = 'LOG' # 'gau', 'LOG'

                # gnn edge function2 for language
                self.G_E_L_S2 = [300*2, feature_size]
                self.G_E_A2   = ['ReLU']
                self.G_E_B2   = bias     #true
                self.G_E_BN2  = bn       #false
                self.G_E_D2   = dropout  #0.3

                # gnn attention mechanism
                self.G_A_L_S = [feature_size, 1]
                self.G_A_A   = ['LeakyReLU']
                self.G_A_B   = bias     #true
                self.G_A_BN  = bn       #false
                self.G_A_D   = dropout  #0.3

                # gnn attention mechanism2 for language
                self.G_A_L_S2 = [feature_size, 1]
                self.G_A_A2   = ['LeakyReLU']
                self.G_A_B2   = bias    #true
                self.G_A_BN2  = bn      #false
                self.G_A_D2   = dropout #0.3
                    
    def save_config(self):
        model_config = {'graph_head':{}, 'graph_node':{}, 'graph_edge':{}, 'graph_attn':{}}
        CONFIG=self.__dict__
        for k, v in CONFIG.items():
            if 'G_H' in k:
                model_config['graph_head'][k]=v
            elif 'G_N' in k:
                model_config['graph_node'][k]=v
            elif 'G_E' in k:
                model_config['graph_edge'][k]=v
            elif 'G_A' in k:
                model_config['graph_attn'][k]=v
            else:
                model_config[k]=v
        
        return model_config


def get_gaussian_filter_1D(kernel_size=3, sigma=2, channels=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()

    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    mean = (kernel_size - 1)/2.
    variance = sigma**2.
    xy_grid = torch.sum((xy_grid[:kernel_size,:kernel_size,:] - mean)**2., dim=-1)

    # Calculate the 1-dimensional gaussian kernel
    gaussian_kernel = (1./((math.sqrt(2.*math.pi)*sigma))) * \
                        torch.exp(-1* (xy_grid[int(kernel_size/2)]) / (2*variance))

    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1)

    padding = 1 if kernel_size==3 else 2 if kernel_size == 5 else 0
    gaussian_filter = nn.Conv1d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels,
                                bias=False, padding=padding)
    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False 
    return gaussian_filter


def get_laplaceOfGaussian_filter_1D(kernel_size=3, sigma=2, channels=3):
    
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    mean = (kernel_size - 1)/2.

    used_sigma = sigma
    # Calculate the 2-dimensional gaussian kernel which is
    log_kernel = (-1./(math.pi*(used_sigma**4))) \
                  * (1-(torch.sum((xy_grid[int(kernel_size/2)] - mean)**2., dim=-1) / (2*(used_sigma**2)))) \
                  * torch.exp(-torch.sum((xy_grid[int(kernel_size/2)] - mean)**2., dim=-1) / (2*(used_sigma**2)))
    
    # Make sure sum of values in gaussian kernel equals 1.
    log_kernel = log_kernel / torch.sum(log_kernel)
    log_kernel = log_kernel.view(1, 1, kernel_size)
    log_kernel = log_kernel.repeat(channels, 1, 1)

    padding = 1 if kernel_size==3 else 2 if kernel_size == 5 else 0
    log_filter = nn.Conv1d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels,
                                bias=False, padding=padding)
    log_filter.weight.data = log_kernel
    log_filter.weight.requires_grad = False
    return log_filter


class Identity(nn.Module):
    '''
    Identity class activation layer
    x = x
    '''
    def __init__(self):
        super(Identity,self).__init__()

    def forward(self, x):
        return x

def get_activation(name):
    '''
    get_activation sub-function
    argument: activatoin name (eg. ReLU, Identity, LeakyReLU)
    '''
    if name=='ReLU': return nn.ReLU(inplace=True)
    elif name=='Identity': return Identity()
    elif name=='LeakyReLU': return nn.LeakyReLU(0.2,inplace=True)
    else: assert(False), 'Not Implemented'
    #elif name=='Tanh': return nn.Tanh()
    #elif name=='Sigmoid': return nn.Sigmoid()

class MLP(nn.Module):
    '''
    Args:
        layer_sizes: a list, [1024,1024,...]
        activation: a list, ['ReLU', 'Tanh',...]
        bias : bool
        use_bn: bool
        drop_prob: default is None, use drop out layer or not
    '''
    def __init__(self, layer_sizes, activation, bias=True, use_bn=False, drop_prob=None):
        super(MLP, self).__init__()
        self.bn = use_bn
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes)-1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i+1], bias=bias)
            activate = get_activation(activation[i])
            block = nn.Sequential(OrderedDict([(f'L{i}', layer), ]))
            
            # !NOTE:# Actually, it is inappropriate to use batch-normalization here
            if use_bn:                                  
                bn = nn.BatchNorm1d(layer_sizes[i+1])
                block.add_module(f'B{i}', bn)
            
            # batch normalization is put before activation function 
            block.add_module(f'A{i}', activate)

            # dropout probablility
            if drop_prob:
                block.add_module(f'D{i}', nn.Dropout(drop_prob))
            
            self.layers.append(block)
    
    def forward(self, x):
        for layer in self.layers:
            # !NOTE: sometime the shape of x will be [1,N], and we cannot use batch-normailzation in that situation
            if self.bn and x.shape[0]==1:
                x = layer[0](x)
                x = layer[:-1](x)
            else:
                x = layer(x)
        return x


class H_H_EdgeApplyModule(nn.Module): #human to human edge
    '''
        init    : config, multi_attn 
        forward : edge
    '''
    def __init__(self, CONFIG, multi_attn=False, use_cbs = False):
        super(H_H_EdgeApplyModule, self).__init__()
        self.use_cbs = use_cbs
        if use_cbs:
            self.init_std = CONFIG.G_E_c_std 
            self.cbs_std = CONFIG.G_E_c_std
            self.cbs_std_factor = CONFIG.G_E_c_std_factor
            self.cbs_epoch = CONFIG.G_E_c_epoch
            self.cbs_kernel_size = CONFIG.G_E_c_kernel_size
            self.cbs_filter = CONFIG.G_E_c_filter
        
        self.edge_fc = MLP(CONFIG.G_E_L_S, CONFIG.G_E_A, CONFIG.G_E_B, CONFIG.G_E_BN, CONFIG.G_E_D)
        self.edge_fc_lang = MLP(CONFIG.G_E_L_S2, CONFIG.G_E_A2, CONFIG.G_E_B2, CONFIG.G_E_BN2, CONFIG.G_E_D2)
    
    def forward(self, edge):
        feat = torch.cat([edge.src['n_f'], edge.data['s_f'], edge.dst['n_f']], dim=1)
        feat_lang = torch.cat([edge.src['word2vec'], edge.dst['word2vec']], dim=1)
        if self.use_cbs:
            feat = self.kernel1(feat[:,None,:])
            feat = torch.squeeze(feat, 1)
        e_feat = self.edge_fc(feat)
        e_feat_lang = self.edge_fc_lang(feat_lang)
  
        return {'e_f': e_feat, 'e_f_lang': e_feat_lang}

    def get_new_kernels(self, epoch_count):
        if self.use_cbs:
            if epoch_count == 0:
                self.cbs_std = self.init_std
                
            if epoch_count % self.cbs_epoch == 0 and epoch_count is not 0:
                self.cbs_std *= self.cbs_std_factor
            
            if (self.cbs_filter == 'gau'): 
                self.kernel1 = get_gaussian_filter_1D(kernel_size=self.cbs_kernel_size, sigma= self.cbs_std, channels= 1)
            elif (self.cbs_filter == 'LOG'): 
                self.kernel1 = get_laplaceOfGaussian_filter_1D(kernel_size=self.cbs_kernel_size, sigma= self.cbs_std, channels= 1)


class H_NodeApplyModule(nn.Module): #human node
    '''
        init    : config
        forward : node
    '''
    def __init__(self, CONFIG):
        super(H_NodeApplyModule, self).__init__()
        self.node_fc = MLP(CONFIG.G_N_L_S, CONFIG.G_N_A, CONFIG.G_N_B, CONFIG.G_N_BN, CONFIG.G_N_D)
        self.node_fc_lang = MLP(CONFIG.G_N_L_S2, CONFIG.G_N_A2, CONFIG.G_N_B2, CONFIG.G_N_BN2, CONFIG.G_N_D2)
    
    def forward(self, node):
        feat = torch.cat([node.data['n_f'], node.data['z_f']], dim=1)
        feat_lang = torch.cat([node.data['word2vec'], node.data['z_f_lang']], dim=1)
        n_feat = self.node_fc(feat)
        n_feat_lang = self.node_fc_lang(feat_lang)

        return {'new_n_f': n_feat, 'new_n_f_lang': n_feat_lang}


class E_AttentionModule1(nn.Module): #edge attention
    '''
        init    : config
        forward : edge
    '''
    def __init__(self, CONFIG):
        super(E_AttentionModule1, self).__init__()
        self.attn_fc = MLP(CONFIG.G_A_L_S, CONFIG.G_A_A, CONFIG.G_A_B, CONFIG.G_A_BN, CONFIG.G_A_D)
        self.attn_fc_lang = MLP(CONFIG.G_A_L_S2, CONFIG.G_A_A2, CONFIG.G_A_B2, CONFIG.G_A_BN2, CONFIG.G_A_D2)

    def forward(self, edge):
        a_feat = self.attn_fc(edge.data['e_f'])
        a_feat_lang = self.attn_fc_lang(edge.data['e_f_lang'])
        return {'a_feat': a_feat, 'a_feat_lang': a_feat_lang}


class GNN(nn.Module):
    '''
        init    : config, multi_attn, diff_edge
        forward : g, h_node, o_node, h_h_e_list, o_o_e_list, h_o_e_list, pop_features
    '''
    def __init__(self, CONFIG, multi_attn=False, diff_edge=True, use_cbs = False):
        super(GNN, self).__init__()
        self.diff_edge = diff_edge # false
        self.apply_h_h_edge = H_H_EdgeApplyModule(CONFIG, multi_attn, use_cbs)
        self.apply_edge_attn1 = E_AttentionModule1(CONFIG)  
        self.apply_h_node = H_NodeApplyModule(CONFIG)

    def _message_func(self, edges):
        return {'nei_n_f': edges.src['n_f'], 'nei_n_w': edges.src['word2vec'], 'e_f': edges.data['e_f'], 'e_f_lang': edges.data['e_f_lang'], 'a_feat': edges.data['a_feat'], 'a_feat_lang': edges.data['a_feat_lang']}

    def _reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['a_feat'], dim=1)
        alpha_lang = F.softmax(nodes.mailbox['a_feat_lang'], dim=1)

        z_raw_f = nodes.mailbox['nei_n_f']+nodes.mailbox['e_f']
        z_f = torch.sum( alpha * z_raw_f, dim=1)

        z_raw_f_lang = nodes.mailbox['nei_n_w']
        z_f_lang = torch.sum(alpha_lang * z_raw_f_lang, dim=1)
         
        # we cannot return 'alpha' for the different dimension 
        if self.training or validation: return {'z_f': z_f, 'z_f_lang': z_f_lang}
        else: return {'z_f': z_f, 'z_f_lang': z_f_lang, 'alpha': alpha, 'alpha_lang': alpha_lang}

    def forward(self, g, h_node, o_node, h_h_e_list, o_o_e_list, h_o_e_list, pop_feat=False):
        
        g.apply_edges(self.apply_h_h_edge, g.edges())
        g.apply_edges(self.apply_edge_attn1)
        g.update_all(self._message_func, self._reduce_func)
        g.apply_nodes(self.apply_h_node, h_node+o_node)

        # !NOTE:PAY ATTENTION WHEN ADDING MORE FEATURE
        g.ndata.pop('n_f')
        g.ndata.pop('word2vec')

        g.ndata.pop('z_f')
        g.edata.pop('e_f')
        g.edata.pop('a_feat')

        g.ndata.pop('z_f_lang')
        g.edata.pop('e_f_lang')
        g.edata.pop('a_feat_lang')


class GRNN(nn.Module):
    '''
    init: 
        config, multi_attn, diff_edge
    forward: 
        batch_graph, batch_h_node_list, batch_obj_node_list,
        batch_h_h_e_list, batch_o_o_e_list, batch_h_o_e_list,
        features, spatial_features, word2vec,
        valid, pop_features, initial_features
    '''
    def __init__(self, CONFIG, multi_attn=False, diff_edge=True, use_cbs = False):
        super(GRNN, self).__init__()
        self.multi_attn = multi_attn #false
        self.gnn = GNN(CONFIG, multi_attn, diff_edge, use_cbs)

    def forward(self, batch_graph, batch_h_node_list, batch_obj_node_list, batch_h_h_e_list, batch_o_o_e_list, batch_h_o_e_list, feat, spatial_feat, word2vec, valid=False, pop_feat=False, initial_feat=False):
        
        # !NOTE: if node_num==1, there will be something wrong to forward the attention mechanism
        global validation 
        validation = valid

        # initialize the graph with some datas
        batch_graph.ndata['n_f'] = feat           # node: features 
        batch_graph.ndata['word2vec'] = word2vec  # node: words
        batch_graph.edata['s_f'] = spatial_feat   # edge: spatial features

        try:
            self.gnn(batch_graph, batch_h_node_list, batch_obj_node_list, batch_h_h_e_list, batch_o_o_e_list, batch_h_o_e_list)
        except Exception as e:
            print(e)


class Predictor(nn.Module):
    '''
    init    : config
    forward : edge
    '''
    def __init__(self, CONFIG):
        super(Predictor, self).__init__()
        self.classifier = MLP(CONFIG.G_ER_L_S, CONFIG.G_ER_A, CONFIG.G_ER_B, CONFIG.G_ER_BN, CONFIG.G_ER_D)
        self.sigmoid = nn.Sigmoid()

    def forward(self, edge):
        feat = torch.cat([edge.dst['new_n_f'], edge.dst['new_n_f_lang'], edge.data['s_f'], edge.src['new_n_f_lang'], edge.src['new_n_f']], dim=1)
        pred = self.classifier(feat)
        # if the criterion is BCELoss, you need to uncomment the following code
        # output = self.sigmoid(output)
        return {'pred': pred}


class AGRNN(nn.Module):
    '''
    init    : 
        feature_type, bias, bn, dropout, multi_attn, layer, diff_edge
        
    forward : 
        node_num, features, spatial_features, word2vec, roi_label,
        validation, choose_nodes, remove_nodes
    '''
    def __init__(self, bias=True, bn=True, dropout=None, multi_attn=False, layer=1, diff_edge=True, use_cbs = False):
        super(AGRNN, self).__init__()
 
        self.multi_attn = multi_attn # false
        self.layer = layer           # 1 layer
        self.diff_edge = diff_edge   # false
        
        self.CONFIG1 = CONFIGURATION(layer=1, bias=bias, bn=bn, dropout=dropout, multi_attn=multi_attn)

        self.grnn1 = GRNN(self.CONFIG1, multi_attn=multi_attn, diff_edge=diff_edge, use_cbs = use_cbs)
        self.edge_readout = Predictor(self.CONFIG1)
        
    def _collect_edge(self, node_num, roi_label, node_space, diff_edge):
        '''
        arguments: node_num, roi_label, node_space, diff_edge
        '''
        
        # get human nodes && object nodes
        h_node_list = np.where(roi_label == 0)[0]
        obj_node_list = np.where(roi_label != 0)[0]
        edge_list = []
        
        h_h_e_list = []
        o_o_e_list = []
        h_o_e_list = []
        
        readout_edge_list = []
        readout_h_h_e_list = []
        readout_h_o_e_list = []
        
        # get all edge in the fully-connected graph, edge_list, For node_num = 2, edge_list = [(0, 1), (1, 0)]
        for src in range(node_num):
            for dst in range(node_num):
                if src == dst:
                    continue
                else:
                    edge_list.append((src, dst))
        
        # readout_edge_list, get corresponding readout edge in the graph
        src_box_list = np.arange(roi_label.shape[0])
        for dst in h_node_list:
            # if dst == roi_label.shape[0]-1:
            #    continue
            # src_box_list = src_box_list[1:]
            for src in src_box_list:
                if src not in h_node_list:
                    readout_edge_list.append((src, dst))
        
        # readout h_h_e_list, get corresponding readout h_h edges && h_o edges
        temp_h_node_list = h_node_list[:]
        for dst in h_node_list:
            if dst == h_node_list.shape[0]-1:
                continue
            temp_h_node_list = temp_h_node_list[1:]
            for src in temp_h_node_list:
                if src == dst: continue
                readout_h_h_e_list.append((src, dst))

        # readout h_o_e_list
        readout_h_o_e_list = [x for x in readout_edge_list if x not in readout_h_h_e_list]

        # add node space to match the batch graph
        h_node_list = (np.array(h_node_list)+node_space).tolist()
        obj_node_list = (np.array(obj_node_list)+node_space).tolist()
        
        h_h_e_list = (np.array(h_h_e_list)+node_space).tolist() #empty no diff_edge
        o_o_e_list = (np.array(o_o_e_list)+node_space).tolist() #empty no diff_edge
        h_o_e_list = (np.array(h_o_e_list)+node_space).tolist() #empty no diff_edge

        readout_h_h_e_list = (np.array(readout_h_h_e_list)+node_space).tolist()
        readout_h_o_e_list = (np.array(readout_h_o_e_list)+node_space).tolist()   
        readout_edge_list = (np.array(readout_edge_list)+node_space).tolist()

        return edge_list, h_node_list, obj_node_list, h_h_e_list, o_o_e_list, h_o_e_list, readout_edge_list, readout_h_h_e_list, readout_h_o_e_list
    
    def _build_graph(self, node_num, roi_label, node_space, diff_edge):
        '''
        Declare graph, add_nodes, collect edges, add_edges
        '''
        graph = dgl.DGLGraph()
        graph.add_nodes(node_num)

        edge_list, h_node_list, obj_node_list, h_h_e_list, o_o_e_list, h_o_e_list, readout_edge_list, readout_h_h_e_list, readout_h_o_e_list = self._collect_edge(node_num, roi_label, node_space, diff_edge)
        src, dst = tuple(zip(*edge_list))
        graph.add_edges(src, dst)   # make the graph bi-directional

        return graph, h_node_list, obj_node_list, h_h_e_list, o_o_e_list, h_o_e_list, readout_edge_list, readout_h_h_e_list, readout_h_o_e_list

    def forward(self, node_num=None, feat=None, spatial_feat=None, word2vec=None, roi_label=None, validation=False, choose_nodes=None, remove_nodes=None):
        
        batch_graph, batch_h_node_list, batch_obj_node_list, batch_h_h_e_list, batch_o_o_e_list, batch_h_o_e_list, batch_readout_edge_list, batch_readout_h_h_e_list, batch_readout_h_o_e_list = [], [], [], [], [], [], [], [], []
        node_num_cum = np.cumsum(node_num) # !IMPORTANT
        
        for i in range(len(node_num)):
            # set node space
            node_space = 0
            if i != 0:
                node_space = node_num_cum[i-1]
            graph, h_node_list, obj_node_list, h_h_e_list, o_o_e_list, h_o_e_list, readout_edge_list, readout_h_h_e_list, readout_h_o_e_list = self._build_graph(node_num[i], roi_label[i], node_space, diff_edge=self.diff_edge)
            
            # updata batch
            batch_graph.append(graph)
            batch_h_node_list += h_node_list
            batch_obj_node_list += obj_node_list
            
            batch_h_h_e_list += h_h_e_list
            batch_o_o_e_list += o_o_e_list
            batch_h_o_e_list += h_o_e_list
            
            batch_readout_edge_list += readout_edge_list
            batch_readout_h_h_e_list += readout_h_h_e_list
            batch_readout_h_o_e_list += readout_h_o_e_list
        
        batch_graph = dgl.batch(batch_graph)
        
        # GRNN
        self.grnn1(batch_graph, batch_h_node_list, batch_obj_node_list, batch_h_h_e_list, batch_o_o_e_list, batch_h_o_e_list, feat, spatial_feat, word2vec, validation, initial_feat=True)
        batch_graph.apply_edges(self.edge_readout, tuple(zip(*(batch_readout_h_o_e_list+batch_readout_h_h_e_list))))
        
        if self.training or validation:
            # !NOTE: cannot use "batch_readout_h_o_e_list+batch_readout_h_h_e_list" because of the wrong order
            return batch_graph.edges[tuple(zip(*batch_readout_edge_list))].data['pred']
        else:
            return batch_graph.edges[tuple(zip(*batch_readout_edge_list))].data['pred'], \
                   batch_graph.nodes[batch_h_node_list].data['alpha'], \
                   batch_graph.nodes[batch_h_node_list].data['alpha_lang'] 
