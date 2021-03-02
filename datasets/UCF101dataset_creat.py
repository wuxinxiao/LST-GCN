"""
Created on May 31 , 2019

@author: Wang Ruiqi

edge feature: the combined bbox of two nodes

Description of the file.
node feature and edge feature both add to the graph
return scene feature


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
def edge_intersection(node1_position, node2_position):
    x1, y1, x2, y2 = [_ for _ in node1_position]
    x1_, y1_, x2_, y2_ = [_ for _ in node2_position]
    # Main diagonal
    if x1_<=x1<=x2_ and y1_<=y1<=y2_:
        if x1<=x2_<=x2 and y1<=y2_<=y2:
            return x1, y1, x2_, y2_
        else:
            return x1, y1, x2, y2
    elif x1_<=x2<=x2_ and y1_<=y2<=y2_:
        if x1<=x1_<=x2 and y1<=y1_<=y2:
            return x1_, y1_, x2, y2
        else:
            return x1, y1, x2, y2
    # Anti-diagonal
    x1, y1, x2, y2 = x2, y1, x1, y2
    x1_, y1_, x2_, y2_ = x2_, y1_, x1_, y2_
    if x2_<x1<x1_ and y1_<y1<y2_:
        if x2<x2_<x1 and y1<y2_<y2:
            return x2_, y1, x1, y2_
        else:
            return x2, y1, x1, y2
    elif x2_<x2<x1_ and y1_<y2<y2_:
        if x2<x1_<x1 and y1<y1_<y2:
            return x2, y1_, x1_, y2
        else:
            return x2, y1, x1, y2
    # No Intersection
    return None

def edge_union(node1_position, node2_position):
    x1, y1, x2, y2 = [_ for _ in node1_position]
    x1_, y1_, x2_, y2_ = [_ for _ in node2_position]
    return min(x1,x1_), min(y1,y1_), max(x2,x2_), max(y2,y2_)




def get_feature(feat_i,anno_file,feat_dir):

    
    feat_category = feat_dir.split('/')[-2]
    #print(feat_category)
    
    '''3DResNext101_rgb'''
    if  feat_category=='3DResNext101Feat-V3':
        #anno_file_ = anno_file.replace('HandstandPushups', 'HandStandPushups')
        video = anno_file.split('/')[-3] + '/' + anno_file.split('/')[-2]
        feat_path = os.path.join(feat_dir, video)
        rgb_feature = np.load(os.path.join(feat_path, feat_i))
        feature = np.squeeze(rgb_feature)
        #print (rgb_feature.shape)
    if  feat_category=='3DResNext101Feat-V3.1':
        #anno_file_ = anno_file.replace('HandstandPushups', 'HandStandPushups')
        video = anno_file.split('/')[-3] + '/' + anno_file.split('/')[-2]
        feat_path = os.path.join(feat_dir, video)
        rgb_feature = np.load(os.path.join(feat_path, feat_i))
        feature = np.squeeze(rgb_feature)
        #print (rgb_feature.shape)
    if  feat_category=='3DResNext101_Feat_AVG':
        #anno_file_ = anno_file.replace('HandstandPushups', 'HandStandPushups')
        video = anno_file.split('/')[-3] + '/' + anno_file.split('/')[-2]
        feat_path = os.path.join(feat_dir, video)
        rgb_feature = np.load(os.path.join(feat_path, feat_i))
        feature = np.squeeze(rgb_feature)
        #print (rgb_feature.shape)


    '''ResNet-18_rgb'''   
    if feat_category=='UCF101_Res18_Feat':
        anno_file_ = anno_file.replace('HandstandPushups', 'HandStandPushups')
        video = anno_file.split('/')[-3] + '/' + anno_file.split('/')[-2]
        feat_path = os.path.join(feat_dir, video)
        feature = np.load(os.path.join(feat_path, feat_i))
        #rgb_feature = np.zeros((512, 7, 7))
        #img_feature = np.concatenate((flow_feature, rgb_feature), 0)
        #del flow_feature
        #del rgb_feature
        #return img_feature
    
    if feat_category=='UCF101_Res50_Feat_imagenet':
        anno_file_ = anno_file.replace('HandstandPushups', 'HandStandPushups')
        video = anno_file.split('/')[-3] + '/' + anno_file.split('/')[-2]
        feat_path = os.path.join(feat_dir, video)
        feature = np.load(os.path.join(feat_path, feat_i))
        
    if feat_category=='UCF101_Res50_Feat_imagenet_fintuned':
        anno_file_ = anno_file.replace('HandstandPushups', 'HandStandPushups')
        video = anno_file.split('/')[-3] + '/' + anno_file.split('/')[-2]
        feat_path = os.path.join(feat_dir, video)
        feature = np.load(os.path.join(feat_path, feat_i))
        
    '''BNInception_rgb'''
    if feat_category=='BNINc_RGB_Feat_last2-V2':
        video = anno_file.split('/')[-3] + '/' + anno_file.split('/')[-2]
        feat_path = os.path.join(feat_dir, video)
        rgb_feature = np.load(os.path.join(feat_path, feat_i))
        feature = np.squeeze(rgb_feature)


    '''ResNet-18_flow'''
    if feat_category =='flow_images':
        anno_file_ = anno_file.replace('HandstandPushups', 'HandStandPushups')
        video = anno_file_.split('/')[-2][:-4]
        feat_path = os.path.join(feat_dir, video)
        flow_feature = np.load(os.path.join(feat_path, feat_i))
        feature = flow_feature
    
    if feat_category =='flow_images_res50':
        anno_file_ = anno_file.replace('HandstandPushups', 'HandStandPushups')
        video = anno_file_.split('/')[-2][:-4]
        feat_path = os.path.join(feat_dir, video)
        flow_feature = np.load(os.path.join(feat_path, feat_i))
        feature = flow_feature

    '''BNInception_flow'''
    if feat_category == 'BNINc_Flow_Feat_last2-V2' or feat_category == 'BNINc_Flow_Feat_AVG':
        video = anno_file.split('/')[-3] + '/'+anno_file.split('/')[-2][:-4]
        feat_path = os.path.join(feat_dir, video)
        flow_feature = np.load(os.path.join(feat_path, feat_i))
        feature = np.squeeze(flow_feature)  
    
    return feature


def sample(files, num):
    files.sort()
    sampled = np.array(np.linspace(0, len(files)-1+1e-3, num), dtype=np.int)
    '''
    print(list(np.array(files)[sampled]))
    exit()
    '''
    return list(np.array(files)[sampled])

def data_sampling(img_list, rgb_dir, flow_dir, video_name,sampling_num):
    '''
    sample 10 samplers for each video
    :param list: video_frame_dir, rgb_feat_dir, flow_feat_dir, video_name 
    :return: sampling img_index, rgb_feat_index, flo_feat_index without their root directory
    '''
    video_frame_path = img_list + video_name
    rgb_feat_path = rgb_dir + video_name
    #print(video_name)
    if rgb_dir==flow_dir:
        flow_feat_path = rgb_feat_path
    else:
        flow_feat_path = flow_dir + video_name.split('/')[-1].split('.')[0].replace('HandstandPushups', 'HandStandPushups')
    #print(flow_feat_path)
    imgList = os.listdir(video_frame_path)
    img_index = sample(imgList,sampling_num)

    rgbList = os.listdir(rgb_feat_path)
    rgb_index = sample(rgbList,sampling_num)

    flowList = os.listdir(flow_feat_path)
    flow_index = sample(flowList,sampling_num)
    
    return img_index, rgb_index, flow_index

def max_node_num(anno_file_list):
    node_num = 1
    for anno_file in anno_file_list:
        fr = open(anno_file)
        anno = pickle.load(fr)
        annos = anno["all_bboxes"]
        if len(annos)>node_num:
            node_num = len(annos)
    return node_num

def read_frame_bbox_feat(frame_bbox, frame_feat):
    #def get_feature_map_coord(bbox, n_grid, ignore_small_bbox=False):
    def get_feature_map_coord(bbox, n_grid_w,n_grid_h, ignore_small_bbox=False):
        # n_grid: num of spatial grid of the feature map
        # bbox: j1, i2, j2, i2, h, w, conf. List or ndArray or sth like that.
        def iou(l1, r1, l2, r2):
            # calculate iou between two line segments, along axis i or j.
            _l = max(l1, l2)
            _r = min(r1, r2)
            # why add 1? Avoid overflow??
            inter = max(0, _r - _l + 1)
            union = (r1 - l1 + 1) + (r2 - l2 + 1) - inter
            return inter / union
        # j1, i2, j2, i2, h, w, conf = bbox
        # j: column index; i: row index;
        j1, i1, j2, i2, h, w = bbox[:6]
        # a grid of feat map correspond to (grid_h*grid_w) pixels
        grid_h = float(h) / n_grid_h
        grid_w = float(w) / n_grid_w

        # row ranges, col ranges
        valid_rows, valid_cols = [], []
        row_iou, col_iou = [], []

        '''resnet feature is 512*7*7'''
        '''
        for i in range(n_grid):
            # why 0.1?**********************
            grid_start_row, grid_center_row, grid_end_row = (round(_ * grid_h) for _ in (i, i + 0.5, i + 0.1))
            grid_start_col, grid_center_col, grid_end_col = (round(_ * grid_w) for _ in (i, i + 0.5, i + 0.1))
            # the bbox covers some grids in the pixel axis (len(valid_rows) * len(valid_cols)) projected from the feature map.
            if i1 <= grid_center_row <= i2:
                valid_rows.append(i)
            if j1 <= grid_center_col <= j2:
                valid_cols.append(i)
            row_iou.append(iou(grid_start_row, grid_end_row, i1, i2))
            col_iou.append(iou(grid_start_col, grid_end_col, j1, j2))
        '''

        for i in range(n_grid_h):
            # why 0.1?**********************
            grid_start_row, grid_center_row, grid_end_row = (round(_ * grid_h) for _ in (i, i + 0.5, i + 0.1))
            # the bbox covers some grids in the pixel axis (len(valid_rows) * len(valid_cols)) projected from the feature map.
            if i1 <= grid_center_row <= i2:
                valid_rows.append(i)
       
            row_iou.append(iou(grid_start_row, grid_end_row, i1, i2))
           

        for i in range(n_grid_w):
            # why 0.1?**********************
           
            grid_start_col, grid_center_col, grid_end_col = (round(_ * grid_w) for _ in (i, i + 0.5, i + 0.1))
            # the bbox covers some grids in the pixel axis (len(valid_rows) * len(valid_cols)) projected from the feature map.
            if j1 <= grid_center_col <= j2:
                valid_cols.append(i)
            col_iou.append(iou(grid_start_col, grid_end_col, j1, j2))

        if len(valid_rows) == 0 or len(valid_cols) == 0:
            if ignore_small_bbox:
                return None
            else:
                valid_rows.append(np.argmax(row_iou))
                valid_cols.append(np.argmax(col_iou))
        # return coord of feature map 
        # why add 1?
        return min(valid_rows), max(valid_rows) + 1, min(valid_cols), max(valid_cols) + 1
    """
    :param bbox: list of [j1, i1, j2, i2, h, w, conf]
    :param frame: ? * 7 * 7 or ? * 8 * 8
    :return:
    """
    #assert frame_feat.shape[-1] == frame_feat.shape[-2], '{}'.format(frame_feat.shape)
    #n_grid = frame_feat.shape[-1]
    n_grid_w = frame_feat.shape[-1]
    n_grid_h = frame_feat.shape[-2]

    all_bbox_feat = []
    for bbox in frame_bbox:
        j1, i1, j2, i2, h, w = bbox[:6]
        #coord = get_feature_map_coord(bbox, n_grid)
        coord = get_feature_map_coord(bbox, n_grid_w,n_grid_h)
        if coord is None:
            all_bbox_feat.append(None)
            continue
        r1, r2, c1, c2 = coord
        bbox_feat = frame_feat[:, r1:r2, c1:c2]
        # Spatial Avg Pooling
        bbox_feat = np.average(bbox_feat.reshape(bbox_feat.shape[0], -1), -1)
        all_bbox_feat.append(bbox_feat)
    return np.array(all_bbox_feat)
        

def create_nodes(anno_file, max_node_num, node_feat_dim, nothing_detected,feat_i,rgb_feat_dir):
    '''
    :param img_file: an image file
    :param anno_file: detected result of the image,
                    an array concat [bbox_coordinate(4),bbox_score(1),feature(1024),class_label(1)]
    :return: nodes in the image
    '''
    #img_feature,tem_feature = get_feature(anno_file, i)  #for rgb and flow feature[512,7,7]
    #del tem_feature

    img_feature = get_feature(feat_i,anno_file,rgb_feat_dir)  #for 3DResNext101 feature[2048, 4, 5]

    if  nothing_detected:
        img_feature = img_feature.mean(-1).mean(-1)
        n_nodes_img = 5
        node_features = np.zeros((max_node_num, node_feat_dim))
        nodes_preserve = [0 for _ in range(5)]
        for i in range(max_node_num):
            node_features[i, :] = img_feature
        return nodes_preserve, n_nodes_img, node_features
    # Lin add 'rb' due to python3 runtime error, on Sept. 1st
    
    fr = open(anno_file, 'rb')
    #fr = open(anno_file)
    anno = pickle.load(fr)
    #annos = anno["all_bboxes"]
    annos = np.array(anno)
    
    bboxes = annos[:, :4]
    frame_bbox = np.concatenate((bboxes, np.ones((len(bboxes),1))*240, np.ones((len(bboxes),1))*320), -1)
    node_score = annos[:, 4]
    # Lin comment
    #node_feature = annos[:, 5:-1]
    node_feature = read_frame_bbox_feat(frame_bbox, img_feature)
    
    
    #node_feature = [img_feature for _ in max_node_num]
    node_label = annos[:, -1] #for something-something

    anno_node_features = np.zeros((len(annos), node_feat_dim))
    node_features = np.zeros((max_node_num, node_feat_dim))
    nodes = []
    for i in range(len(annos)):
        #node = dict(node_features=node_features[i], node_position=bboxes[i,:], node_label=node_label[i])
        node = dict(node_position=bboxes[i,:], node_score=node_score[i], node_label=node_label[i], node_state=node_feature[i])
        #*************wang**************#
        #anno_node_features[i] = node_feature[i] 
        #*******************************#
        nodes.append(node)

    if len(nodes) > max_node_num:
        ## New method for dealing with too many boxes
        # delete bbox with an high IoU with other bboxes
        iou_mat = np.zeros((len(nodes), len(nodes)))
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if i<=j: continue
                node1, node2 = nodes[i]['node_position'], nodes[j]['node_position']
                area1 = (node1[2]-node1[0]) * (node1[3]-node1[1])
                iou = edge_intersection(node1, node2)
                if iou is None:
                    iou_mat[i][j] = 0
                    continue
                iou_area = (iou[2]-iou[0]) * (iou[3]-iou[1])
                iou_mat[i][j] = iou_area / area1
        iou_vec = iou_mat.reshape(-1)
        to_remove = []
        for i in range(len(nodes) - max_node_num):
            index = np.argmax(iou_vec)
            row = index // len(nodes)
            col = index - (index//len(nodes))*len(nodes)
            '''
            # wrong version
            if nodes[row]['node_score'] < nodes[col]['node_score']:
                iou_vec[row*len(nodes): row*(len(nodes)+1)] = 0
                iou_vec[row::len(nodes)] = 0
                to_remove.append(row)
            else:
                iou_vec[col*len(nodes): col*(len(nodes)+1)] = 0
                iou_vec[col::len(nodes)] = 0
                to_remove.append(col)
            '''
            if nodes[row]['node_score'] < nodes[col]['node_score']:
                iou_vec[row*len(nodes): (row+1)*len(nodes)+1] = 0
                iou_vec[row::len(nodes)] = 0
                to_remove.append(row)
            else:
                iou_vec[col*len(nodes): (col+1)*len(nodes)+1] = 0
                iou_vec[col::len(nodes)] = 0
                to_remove.append(col)
        to_remove = sorted(to_remove, reverse=True)
        for i in to_remove: # from high to low
            del nodes[i]
        List = [nodes[_]['node_state'] for _ in range(max_node_num)]
        node_features = np.array(List)
        assert node_features.shape == (max_node_num, node_feat_dim), node_features.shape
        n_nodes_img = max_node_num
        nodes_preserve = nodes


        '''
        ## Old method for dealing with too many bboxes
        #nodes_sort1 = sorted(nodes, key=lambda e: e.__getitem__('node_label'))  #sorted by label
        #*************Lin add 'reverse=True'***************************#
        nodes_sort2 = sorted(nodes, key=lambda e: (e.__getitem__('node_score')), reverse=True)  #sorted by score
        # => sort multiple times: key=tuple[-1] -> key=tuple[-2] -> ... -> key=tuple[0] <=> Priority: key=tuple[0] > key=tuple[1] > ...
        #**************************************************************#
        nodes_preserve = nodes_sort2[: max_node_num]
        #*********************Wang***********************#
        #node_features = anno_node_features[: max_node_num]
        #************************************************#
        #*********************Lin************************#
        List = [nodes_sort2[_]['node_state'] for _ in range(max_node_num)]
        node_features = np.array(List)
        assert node_features.shape == (max_node_num, node_feat_dim), node_features.shape
        #************************************************#
        n_nodes_img = max_node_num
        '''


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

def create_edges(anno_file, nodes, max_node_num, edge_feat_dim,nothing_detected,feat_i,rgb_feat_dir):
    """
    relative position and relative distance (and co-occur possibility) of two detected objects with their concat feature construct an edge
    :param nodes: detected nodes id
    :return: edges and the number of edge

    """

    # Lin comment on Sept. 13
    #edge_feature = anno["imgfeat"]
    #frame_feat, tem_feature = get_feature(anno_file, i)
    #del tem_feature
    frame_feat = get_feature(feat_i,anno_file,rgb_feat_dir) #ResNext101

    if nothing_detected:
     
        
        img_feature =frame_feat.mean(-1).mean(-1)
        
        n_edge_img = max_node_num*max_node_num
        edge_features = np.zeros((n_edge_img, edge_feat_dim))
        edge_list = []
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                edge_type = [i,j]
                edge_state = img_feature
                edge_list.append(dict(edge_type=edge_type, edge_state = edge_state))
                edge_features[i*len(nodes)+j, :] = np.hstack((np.array([0.5,0.5,0,0]), edge_state))

        return edge_list, n_edge_img, edge_features

    # Lin added 'rb' due to python3 runtime error, on Sept. 1st
    fr = open(anno_file, 'rb')
    #fr = open(anno_file)
    anno = pickle.load(fr)

    
    edge_list = []
    #if len(nodes)==0:
    #    return edge_list,0
    n_edge_img = max_node_num*max_node_num
    edge_features = np.zeros((n_edge_img, edge_feat_dim))
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

            assert node2_position[3]-node2_position[1]>0, anno_file
            r_s_w = math.log((node1_position[2]-node1_position[0]+1e-7)/(node2_position[2]-node2_position[0]))
            r_s_h = math.log((node1_position[3]-node1_position[1]+1e-7)/(node2_position[3]-node2_position[1]))
            b_os = [r_p_x,r_p_y,r_s_w,r_s_h]
            b_os = np.array(b_os)
            #feature
            node1_state = nodes[i]['node_state']
            node2_state = nodes[j]['node_state']
            #TODO edge state can have some definitions
            
            # calculate the type of edge
            node1_label =  nodes[i]['node_label']
            node2_label =  nodes[j]['node_label']
            
            '''
            #edge_feature = np.hstack((nodes[i]['node_state'],nodes[j]['node_state']))#TODO change edge_state to the input one
            haha = edge_intersection(node1_position, node2_position)
            if haha:
                frame_bbox = np.concatenate((np.array(haha), [240], [320])).reshape(1, -1)
                edge_feature = read_frame_bbox_feat(frame_bbox, frame_feat).reshape(-1)
            else:
                # AVG Pooling
                edge_feature = (node1_state + node2_state) / 2
                # Max Pooling
                #edge_feature = np.max(np.concatenate((node1_state, node2_state), 0), axis=0)
            '''
            if False:
                frame_bbox = np.concatenate((np.array(haha), [240], [320])).reshape(1, -1)
                edge_feature = read_frame_bbox_feat(frame_bbox, frame_feat).reshape(-1)
            else:
                haha = edge_union(node1_position, node2_position)
                frame_bbox = np.concatenate((np.array(haha), [240], [320])).reshape(1, -1)
                edge_feature = read_frame_bbox_feat(frame_bbox, frame_feat).reshape(-1)


            edge_state = np.hstack((b_os, edge_feature)) #edge feature incorporate relative position
            #edge_state = np.array(edge_state)`
            edge_features[i*len(nodes)+j, :] = edge_state
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

def create_adjacency_matrix_abl(edge_list, n_nodes,max_node_num):     

    """
    :param edges: created edges
    :param n_nodes: number of objects(including person) can be detected
    :param n_edge_types: number of total edges
    :return: initial adjacent matrix for the undirected graph of each frame

    """
    n_edge_types = max_node_num*max_node_num
    
    a = np.zeros([max_node_num, max_node_num * n_edge_types])
    # a = np.zeros([n_nodes, n_nodes])
    #print ('edge_list',edge_list)
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

def create_adjacency_matrix(edge_list, n_nodes,max_node_num, nodes_feature):     # TODO too large,need change afterwards

    """
    :param edges: created edges
    :param n_nodes: #objects(including person) can be detected
    :param n_edge_types: #total edges
    :return: initial adjacent matrix for the direct graph of each frame

    
    a = np.zeros((max_node_num, max_node_num*max_node_num))
    for i in range(n_nodes):
        for j in range(n_nodes):
            a[i,i*n_nodes+j]=1
    return a  #for something-something it is a fully connected graph initially
    """
    a = np.zeros((max_node_num, max_node_num*max_node_num))
    #print(nodes_feature.shape)
    matrix = np.matmul(nodes_feature,nodes_feature.T)
    #print(matrix.shape)
    for i in range(max_node_num):
        a[i][i*max_node_num:(i+1)*max_node_num] =  matrix[i]
    return a
    
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

    def __init__(self, datalist, detect_path, video_frame_dir,rgb_feat_dir,flow_feat_dir, opt):
        self.data = datalist
        self.classNum = opt.classNum
        self.node_feat_size = opt.node_feat_dim
        self.edge_feat_size = opt.edge_feat_dim
        self.rgb_feat_dir = rgb_feat_dir
        self.flow_feat_dir = flow_feat_dir
        self.node_max_num = opt.max_num_node
        self.ggnn = opt.ggnn
        imgs_index_ = []
        rgb_feat_index_ = []
        flow_feat_index_ = []
        ##flos_index_ = []
        detect_list_ = []
        ##imgFeat_list_ = []
        ##floFeat_list_ = []
        video_labels = []
        videos_length = []
        videos_name = []
        #self.data = list()

        # video_lines
        # Lin changed 'rb' to 'r' on Sept. 1st
        lines = open(datalist, 'r')
        for i, line in enumerate(lines):
            line = line.strip()
            #print(line)
            video_name, video_label, video_length = line.split(' ')[0], int(line.split(' ')[1]), int(line.split(' ')[2])
            video_labels.append(video_label)
            videos_length.append(video_length)
            videos_name.append(video_name)
            # detect file(without ext)
            detect_list = detect_path + video_name
            detect_list_.append(detect_list)
            
            # selective img list(without ext)
            video_frame_path = video_frame_dir + video_name
            rgb_feat_path = rgb_feat_dir + video_name
            flow_feat_path = flow_feat_dir + video_name

            img_idx,rgb_feat_idx,flow_feat_idx = data_sampling(video_frame_dir,rgb_feat_dir,flow_feat_dir, video_name,opt.seq_size)
            imgs_index_.append(img_idx)
            rgb_feat_index_.append(rgb_feat_idx)
            flow_feat_index_.append(flow_feat_idx)
            #if i == 50:
                #break
            #imgFeat_list_.append(img_feat_path)
       
        self.imgs_index = imgs_index_
        self.detect_list = detect_list_
        self.rgb_feat_index = rgb_feat_index_
        self.flow_feat_index = flow_feat_index_
        #self.imgFeat_list = imgFeat_list_
        self.video_label = video_labels
        self.video_length = videos_length
        self.video_name = videos_name


    def __getitem__(self, index):
        #line = self.data[index].strip()
        imgInd = self.imgs_index[index]
        rgbInd = self.rgb_feat_index[index]
        flowInd = self.flow_feat_index[index]
        #print(flowInd)
        ##floInd = self.flos_index[index]
        _detect_list =self.detect_list[index]
        ##_imgFeat_list =self.imgFeat_list[index]
        ##_floFeat_list =self.floFeat_list[index]
        video_label =self.video_label[index]
        video_name = self.video_name[index]
        #video_length = self.video_length[index]
        ##imgFeats = []
        ##res3dFeats = []
        adj_matrixes = []
        nodes_features=[]
        edges_features=[]
        scene_features=[]
        #tem_features = []
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
        nothing_detected = False
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

            if (not os.path.exists(os.path.join(_detect_list, imgInd[i][:-4] + '.pkl'))):
                # no object detected in this frame
                if (not os.path.exists(_detect_list)) or len(os.listdir(_detect_list))<1:
                    # no object detected in this video
                    nodes = []
                    detect_file = os.path.join(_detect_list, imgInd[i][:-4] + '.pkl')
                    nothing_detected = True
                else:
                    files_det = os.listdir(_detect_list)
                    files_det.sort()
                    #Use the earliest detected pickle
                    file_instead = os.path.join(_detect_list, files_det[0])
                    # Lin comment on Sept. 12
                    #detect_file = file_instead
                    # Lin add if-else on Sept. 12
                    if i==0:
                        detect_file = file_instead
                    else:
                        pass
                    #detect_files.append(detect_file)
                    #nodes, n_node_img,nodes_feature = create_nodes(file_instead,self.node_feat_size)
            else:
                detect_file = os.path.join(_detect_list, imgInd[i][:-4] + '.pkl')
                #detect_files.append(detect_file)
           
            #nodes, n_node_img,nodes_feature = create_nodes(detect_files[i],video_node_num,self.node_feat_size)
            nodes, n_node_img, nodes_feature = create_nodes(detect_file, video_node_num, self.node_feat_size, nothing_detected,rgbInd[i],self.rgb_feat_dir)
            edges, n_edge_img, edges_feature = create_edges(detect_file, nodes, video_node_num, self.edge_feat_size, nothing_detected,rgbInd[i],self.rgb_feat_dir)
            if self.ggnn:
                print("*****************")
                adj_matrix = create_adjacency_matrix_abl(edges, len(nodes),video_node_num, nodes_feature) #for GGNN
            else:
                adj_matrix = create_adjacency_matrix(edges, len(nodes),video_node_num, nodes_feature)
            
            #rgb_feature,tem_feature = get_feature(detect_file,i)  #two resnet
            #tem_feature = np.average(tem_feature.reshape(tem_feature.shape[0], -1), -1)
            
            ##rgb_feature = get_flow_feature(self.flow_feat_dir,detect_file,i)
            
            flow_feature = get_feature(flowInd[i],detect_file,self.flow_feat_dir)

            scene_feature = np.average(flow_feature.reshape(flow_feature.shape[0], -1), -1)
            #del rgb_feature
            
            



            adj_matrixes.append(adj_matrix)
            nodes_features.append(nodes_feature)
            edges_features.append(edges_feature)
            scene_features.append(scene_feature)
            one_hot_graph_labels = np.zeros(self.classNum+1)
            one_hot_graph_labels[video_label] = 1
            one_hot_graph_labels[-1] = len(imgInd)
            labels.append(one_hot_graph_labels)
            
            #tem_features.append(tem_feature)
        ##imgFeats = np.array(imgFeats)
        #floFeats = np.array(floFeats)
        adj_matrixes = np.array(adj_matrixes, dtype=np.int32)
        nodes_features = np.array(nodes_features, dtype=np.float32)
        edges_features = np.array(edges_features, dtype=np.float32)
        scene_features = np.array(scene_features, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)
        #tem_features =np.array(tem_features, dtype=np.float32)
      
        #return imgFeats, res3dFeats, adj_matrixes, annotations,labels
        ##return nodes_feature, edge_feature, adj_matrixes,labels
        #print("********")
        return edges_features, nodes_features, adj_matrixes, labels, scene_features, video_name

    def __len__(self):
        return len(self.imgs_index)

