import sys
#sys.path.append("./")
import argparse
import random
import os
import time
import numpy as np
import scipy.io as scio
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
#from datasets.dataset_UCF101 import dataset
import UCF101dataset_creat
#import datasets.utils as utils
import config
#import models
#import datasets

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
parser.add_argument('--seq_size', type=int, default=10, help='sequence length of rnn')
parser.add_argument('--classNum', type=int, default=101, help='number of classes')
parser.add_argument('--max_num_node', type=int, default=5, help='number of classes')
parser.add_argument('--subsample_num', type=int, default=10, help='number of classes')
parser.add_argument('--d_pos', type=int, default=256, help='dimension of position')
parser.add_argument('--n_type_node', type=int, default=11, help='#objects+human')
parser.add_argument('--node_feat_dim', type=int, default=512, help='node state size')  #1024 #512 
parser.add_argument('--edge_feat_dim', type=int, default=516, help='edge state size')  #1028  #516
parser.add_argument('--tem_feat_dim', type=int, default=512, help='edge state size')
parser.add_argument('--state_dim', type=int, default=512, help='dim of annotation')
parser.add_argument('--num_bottleneck', type=int, default=256, help='dim of temporal reasoning module')
parser.add_argument('--num_frames', type=int, default=8, help='number of sampled frames in each segment ')
parser.add_argument('--ggnn', action='store_true', help='enables cuda')
opt = parser.parse_args()

def main(opt):
    train_datalist = '/home/mcislab/wangruiqi/IJCV2019/data/ucf101Vid_all_lin.txt'
    #train_datalist = '/home/mcislab/wangruiqi/IJCV2019/data/test.txt'
    save_path = '/media/mcislab/new_ssd/wrq/data/UCF101/res18_rgbflow_same_similiarity/'
    paths = config.Paths()
    train_dataset = UCF101dataset_creat.dataset(train_datalist, paths.detect_root_ucf_mmdet, paths.img_root_ucf,paths.rgb_res18_ucf,paths.rgb_res18_ucf,opt)#rgb_bninc_ucf
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers, drop_last=False)
    print('start to read feature...')
    count =0
    for i, (edge_features, node_features, adj_mat, graph_label,tem_features,video_info) in enumerate(train_dataloader, 0):
        #print (video_info)
        videoinfo = video_info[0]
        category, video_name = videoinfo.split('/')
        edge_features = edge_features.squeeze(0)
        node_features = node_features.squeeze(0)
        adj_mat = adj_mat.squeeze(0)
        graph_label = graph_label.squeeze(0)
        tem_features = tem_features.squeeze(0)
        #print (category)
        #print (video_name)
        #print (edge_features.shape)
        feat_path = os.path.join(save_path,category)
        if not os.path.exists(feat_path):
            os.makedirs(feat_path)
        feat_dict = {
            'edge_features':edge_features,
            'node_features':node_features,
            'adj_mat':adj_mat,
            'graph_label':graph_label,
            'tem_features':tem_features,
            'video_info':video_info
        }
        #print(edge_features.shape)
        print(feat_path+'/'+video_name[:-4]+'.pkl')
        file = open(feat_path+'/'+video_name[:-4]+'.pkl','wb')
        pickle.dump(feat_dict,file)
        count += 1
    print('Has tranformed %d input features. Finished!'%count)
    

if __name__ == "__main__":
    main(opt)
