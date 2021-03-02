"""
Created on May 31 , 2019

@author: Wang Ruiqi

Description of the file.
node feature and edge feature both add to the graph


version3.0

"""

import os
import sys
import time
import pickle
import argparse

import math
import numpy as np
from PIL import Image
from collections import defaultdict

from torch.utils.data import Dataset

import torch
import torch.nn as nn
from torchvision import models
#import torchvision.transforms as transforms
from torch.autograd import Variable
'''Lin commented on Sept. 1st
import pretrainedmodels
import pretrainedmodels.utils as utils
'''
import h5py
import pandas as pd
'''Lin commented on Sept. 1st
import config
'''
def data_sampling(img_list, video_length):
    '''
    sample ten images
    :param list: data_list
    :return: sampling img and flo data list
    '''
    imgList = os.listdir(img_list)
    imgList.sort()
    video_length = len(imgList)
    slt_frms = np.ceil(np.arange(0.1,1.1,0.1) * video_length)

    img = []
    #print slt_frms
    for index, slt_frm in enumerate(slt_frms):
        #img.append(imgList[int(slt_frm)-1])
        #print index
        img.append(imgList[int(slt_frm)-1][:-4])
    return img

def max_node_num(anno_file_list):
    node_num = 1
    for anno_file in anno_file_list:
        fr = open(anno_file)
        anno = pickle.load(fr)
        annos = anno["all_bboxes"]
        if len(annos)>node_num:
            node_num = len(annos)
    return node_num

def create_nodes(anno_file, max_node_num, node_feat_dim):
    '''
    :param img_file: an image file
    :param anno_file: detected result of the image,
                    an array concat [bbox_coordinate(4),bbox_score(1),feature(1024),class_label(1)]
    :return: nodes in the image
    '''
    # Lin add 'rb' due to python3 runtime error, on Sept. 1st
    fr = open(anno_file, 'rb')
    #fr = open(anno_file)
    anno = pickle.load(fr)
    annos = anno["all_bboxes"]
    bboxes = annos[:,:4]
    node_score = annos[:,4]
    node_feature = annos[:, 5:-1]
    #node_label = get_node_label(annos[:,-1])
    node_label = annos[:,-1] #for something-something

    anno_node_features = np.zeros((len(annos), node_feat_dim))
    node_features = np.zeros((max_node_num, node_feat_dim))

    nodes = []
    for i in range(len(annos)):
        #node = dict(node_features=node_features[i], node_position=bboxes[i,:], node_label=node_label[i])
        node = dict(node_position=bboxes[i,:], node_score=node_score, node_label=node_label[i], node_state=node_feature[i])
        #*************wang**************#
        #anno_node_features[i] = node_feature[i]
        #*******************************#
        nodes.append(node)

    if len(nodes) > max_node_num:
        # default: ascending order
        nodes_sort1 = sorted(nodes, key=lambda e: e.__getitem__('node_label'))  #sorted by label
        #*************Lin add 'reverse=True'***************************#
        nodes_sort2 = sorted(nodes_sort1, key=lambda e: (e.__getitem__('node_label'), e.__getitem__('node_score')), reverse=True)  #sorted by score
        # => sort multiple times: key=tuple[-1] -> key=tuple[-2] -> ... -> key=tuple[0] <=> Priority: key=tuple[0] > key=tuple[1] > ...
        #**************************************************************#
        nodes_preserve = nodes_sort2[: max_node_num]
        #*********************Wang***********************#
        #node_features = anno_node_features[: max_node_num]
        #************************************************#
        #*********************Lin************************#
        node_features = np.array([nodes_sort2[_]['node_state'] for _ in range(max_node_num)])
        assert node_features.shape == (max_node_num, node_feat_dim)
        #************************************************#
        n_nodes_img = max_node_num
    else:
        nodes_preserve = nodes
        #*********************Wang***********************#
        #node_features[:len(nodes)] = anno_node_features
        #************************************************#
        #*********************Lin************************#
        node_features[:len(nodes)] = np.array([nodes[_]['node_state'] for _ in range(len(nodes))])
        assert node_features.shape == (max_node_num, node_feat_dim)
        #************************************************#
        n_nodes_img = len(nodes)

    return nodes_preserve, n_nodes_img, node_features

def create_edges(anno_file, nodes, max_node_num, edge_feat_dim):
    """
    relative position and relative distance (and co-occur possibility) of two detected objects with their concat feature construct an edge
    :param nodes: detected nodes id
    :return: edges and the number of edge

    """
    # Lin added 'rb' due to python3 runtime error, on Sept. 1st

    fr = open(anno_file, 'rb')
    #fr = open(anno_file)
    anno = pickle.load(fr)
    edge_feature = anno["imgfeat"]
    edge_list = []
    #if len(nodes)==0:
    #    return edge_list,0
    n_edge_img = max_node_num*max_node_num
    
    edge_features = np.zeros((n_edge_img,edge_feat_dim))

    #for i in range(len(nodes)-1 ):
    #    for j in range(i+1,len(nodes)):
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            # edge_feature = roi_pooling(bbox[node1,node2])

            node1_position = nodes[i]['node_position']
            node2_position = nodes[j]['node_position']
            node1_center = [(node1_position[0]+node1_position[2])/2,(node1_position[1]+node1_position[3])/2]
            node2_center = [(node2_position[0]+node2_position[2])/2,(node2_position[1]+node2_position[3])/2]
            #check_neg = np.array([node1_center[0],node1_center[1],node2_center[0],node2_center[1]])
            #print check_neg
            #if ~((check_neg<0).any()):
            #    print "Node position coordinate should not be negative!"
            #    exit()

            edge_x0 = min(node1_position[0],node2_position[0])
            edge_y0 = min(node1_position[1],node2_position[1])
            edge_x1 = max(node1_position[2],node2_position[2])
            edge_y1 = max(node1_position[3],node2_position[3])
            edge_bbox = np.array([edge_x0,edge_y0,edge_x1,edge_y1])

            #check for math formulation
            #if node2_position[2]-node2_position[0]==0 or 

            # b_os = ((x_o-x_s)/w_s,(y_o-y_s)/h_s,log(w_o/w_s),log(h_o/h_s))  relative position
            r_p_x = abs(node1_center[0]-node2_center[0])/(node2_position[2]-node2_position[0])
            r_p_y = abs(node1_center[1]-node2_center[1])/(node2_position[3]-node2_position[1])
            r_s_w = math.log((node1_position[2]-node1_position[0])/(node2_position[2]-node2_position[0]))
            r_s_h = math.log((node1_position[3]-node1_position[1])/(node2_position[3]-node2_position[1]))
            b_os = [r_p_x,r_p_y,r_s_w,r_s_h]
            b_os = np.array(b_os)
            #feature
            node1_state = nodes[i]['node_state']
            node2_state = nodes[j]['node_state']
            #TODO edge state can have some definitions
            
            # calculate the type of edge
            node1_label =  nodes[i]['node_label']
            node2_label =  nodes[j]['node_label']
            
            #edge_feature = np.hstack((nodes[i]['node_state'],nodes[j]['node_state']))#TODO change edge_state to the input one
            edge_state = np.hstack((b_os, edge_feature)) #edge feature incorporate relative position

            #edge_state = np.array(edge_state)
            #print "i=",i
            #print "j=",j

            edge_features[i*len(nodes)+j,:] = edge_state
            #edge_features[j*(1+len(nodes))+i,:] = edge_state


            #edge_type = [node1_label,node2_label]
            edge_type = [i,j]

            #edge_list.append([edge_type,edge_features,b_os])
            edge_list.append(dict(edge_type=edge_type, edge_state = edge_state))

    return edge_list, n_edge_img, edge_features

def make_init_input(edge_list,n_node, node_feat_size):
    '''
    initial input of the graph actually is the initial hidden state of nodes in the graph, consists of position and feature of the node
    :param n_node:  #total defined nodes
    :param state_dim: #dim of hidden state of each node
    :param n_annotation_dim: #dim of annotation
    :param annotation: annotation made by make_annoataion()
    :return: an array of initial input for the graph, and size is [n_nodes,state_dim]
    '''
    init_input = np.zeros([n_node, node_feat_size])
    #init_input = np.concatenate((annotation, padding), axis=1)
    for edge in edge_list:
        edge_type = edge["edge_type"]
        init_input[int(edge_type[0])-1,:] = edge['node1_state']
        init_input[int(edge_type[1])-1,:] = edge['node2_state']

    return init_input

def create_adjacency_matrix(edge_list, n_nodes,max_node_num):     # TODO too large,need change afterwards

    """
    :param edges: created edges
    :param n_nodes: #objects(including person) can be detected
    :param n_edge_types: #total edges
    :return: initial adjacent matrix for the direct graph of each frame

    """
    a = np.zeros((max_node_num, max_node_num*max_node_num))
    for i in range(n_nodes):
        for j in range(n_nodes):
            a[i,i*n_nodes+j]=1
    return a  #for something-something it is a fully connected graph initially

    '''
    a = np.zeros([n_nodes, n_edge_types])
    
    
    for edge in edge_list:
        edge_type = edge['edge_type']
        #print ("edge type is ",edge_type)
        src_idx = int(edge_type[0])
        tgt_idx = int(edge_type[1])
    
        start = n_nodes-1
        e_type_i = 0
        
        for i in range(src_idx):
            e_type_i = start*i + e_type_i
            start = start - 1
        e_type = e_type_i + tgt_idx -1
        a[src_idx-1][e_type-1] = 1
        
        

        #a[tgt_idx-1][(e_type - 1) * n_nodes + src_idx - 1] =  1
        #a[src_idx-1][(e_type - 1 + n_edge_types) * n_nodes + tgt_idx - 1] =  1
    return a
    '''

class dataset(Dataset):
    features = None

    def __init__(self, datalist, detect_path, img_feat_dir, opt):
        self.data = datalist
        self.classNum = opt.classNum
        self.node_feat_size = opt.node_feat_dim
        self.edge_feat_size = opt.edge_feat_dim
        self.node_max_num = opt.max_num_node
        imgs_index_ = []
        ##flos_index_ = []
        detect_list_ = []
        ##imgFeat_list_ = []
        ##floFeat_list_ = []
        video_labels = []
        videos_length = []

        #self.data = list()

        # video_lines
        # Lin changed 'rb' to 'r' on Sept. 1st
        lines = open(datalist, 'r')
        for i,line in enumerate(lines):
            line = line.strip()
            video_name, video_label, video_length = line.split(' ')[0], int(line.split(' ')[1]), int(line.split(' ')[2])
            video_labels.append(video_label)
            videos_length.append(video_length)

            # detect file(without ext)
            detect_list = detect_path + video_name
            detect_list_.append(detect_list)
            
            # selective img list(without ext)
            img_feat_path = img_feat_dir + video_name
            img_idx = data_sampling(img_feat_path, video_length)
            imgs_index_.append(img_idx)
            
            #if i==50:
                #break
            
            #imgFeat_list_.append(img_feat_path)
       
        self.imgs_index = imgs_index_
        self.detect_list = detect_list_
        #self.imgFeat_list = imgFeat_list_
        self.video_label = video_labels
        self.video_length = videos_length


    def __getitem__(self, index):
        #line = self.data[index].strip()
        imgInd = self.imgs_index[index]
        #print "imgInd", imgInd
        ##floInd = self.flos_index[index]
        _detect_list =self.detect_list[index]
        ##_imgFeat_list =self.imgFeat_list[index]
        ##_floFeat_list =self.floFeat_list[index]
        video_label =self.video_label[index]
        #video_length = self.video_length[index]
        ##imgFeats = []
        ##res3dFeats = []
        adj_matrixes = []
        nodes_features=[]
        edges_features=[]
        labels = []
        '''
        for i in range(len(imgInd)): 
            if not os.path.exists(os.path.join(_detect_list, imgInd[i]+ '.pkl')):
                if not os.path.exists(_detect_list) :
                    nodes = []
                    detect_files = []
                else:
                    files_det = os.listdir(_detect_list)
                    files_det.sort()
                    file_instead = os.path.join(_detect_list, files_det[0])
                    #print file_instead
                    detect_file = file_instead
                    detect_files.append(detect_file)
                   
                    #nodes, n_node_img,nodes_feature = create_nodes(file_instead,self.node_feat_size)
               # print os.path.join(_detect_list, imgInd[i]+ '.pkl')
               # print "no node"
            else:
                detect_file = os.path.join(_detect_list, imgInd[i] + '.pkl')
                detect_files.append(detect_file)

        #video_node_num =  max_node_num(detect_files)      
        '''
        video_node_num = self.node_max_num
        for i in range(len(imgInd)):  # seq_size
            # Create a graph for each image

            ##feat_file = open(os.path.join(_imgFeat_list,imgInd[i] + '.pkl'))
            ##feat = pickle.load(feat_file)
            ##imgFeats.append(feat)

            #feat_file = open(os.path.join(_floFeat_list,floInd[i] + '.pkl'))
            #feat = pickle.load(feat_file)
            #floFeats.append(feat)

            ##res3d_file = h5py.File(os.path.join(_floFeat_list,floInd[i]+'.mat'),'r')
            ##res3d_feat = np.array(res3d_file['feat_res3d'][:])
            ##res3dFeats.append(res3d_feat)
            if not os.path.exists(os.path.join(_detect_list, imgInd[i] + '.pkl')):
                # no object detected in this frame
                if not os.path.exists(_detect_list) :
                    # no object detected in this video
                    nodes = []
                else:
                    files_det = os.listdir(_detect_list)
                    files_det.sort()
                    #Use the earliest detected pickle
                    file_instead = os.path.join(_detect_list, files_det[0])
                    detect_file = file_instead
                    #print file_instead
                    
                    #detect_files.append(detect_file)
                   
                    #nodes, n_node_img,nodes_feature = create_nodes(file_instead,self.node_feat_size)
               # print os.path.join(_detect_list, imgInd[i]+ '.pkl')
               # print "no node"
            else:
                detect_file = os.path.join(_detect_list, imgInd[i] + '.pkl')
                #detect_files.append(detect_file)
           
            #nodes, n_node_img,nodes_feature = create_nodes(detect_files[i],video_node_num,self.node_feat_size)
            nodes, n_node_img, nodes_feature = create_nodes(detect_file, video_node_num, self.node_feat_size)
            '''
            print(type(nodes_feature), len(nodes_feature), nodes_feature[0].shape)
            exit()
            '''
            edges, n_edge_img, edges_feature = create_edges(detect_file, nodes, video_node_num, self.edge_feat_size)
            # print (edges)
          
            #init_input = make_init_input(edges,len(nodes),self.node_feat_size)
            # print (each_frame_graph)
            
            adj_matrix = create_adjacency_matrix(edges, len(nodes),video_node_num)
            # annotation = each_frame_graph[:]['annotation']
            #print "adj_matrix size is ",adj_matrix.shape
            adj_matrixes.append(adj_matrix)
            
            #init_inputs.append(init_input)
            nodes_features.append(nodes_feature)
            edges_features.append(edges_feature)
            one_hot_graph_labels = np.zeros(self.classNum+1)
            one_hot_graph_labels[video_label] = 1
            one_hot_graph_labels[-1] = len(imgInd)
            labels.append(one_hot_graph_labels)

        ##imgFeats = np.array(imgFeats)
        #floFeats = np.array(floFeats)

    
        adj_matrixes = np.array(adj_matrixes)
        nodes_features = np.array(nodes_features)
        '''
        print([nodes_features[_].shape for _ in range(len(nodes_features))])
        exit()
        '''
        edges_features = np.array(edges_features)
        labels = np.array(labels)

        #print adj_matrixes.shape
        #print nodes_features.shape

        #return imgFeats, res3dFeats, adj_matrixes, annotations,labels
        ##return nodes_feature, edge_feature, adj_matrixes,labels
        return edges_features, nodes_features, adj_matrixes, labels

    def __len__(self):
        return len(self.imgs_index)

