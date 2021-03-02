"""
Created on Mar 14 , 2019

@author: Wang Ruiqi

Description of the file.
Using publicly available node feature and edge feature from Ref[1]
[1]Learning human activities and object affordances from RGB-D videos


version3.0

"""


import os
import time
import pickle
import argparse

import numpy as np
import torch.utils.data
import sys
sys.path.append("/home/mcislab/wangruiqi/IJCV2019/datasets/CAD120/")


import cad120_config
import metadata


class CAD120(torch.utils.data.Dataset):
    features = None

    def __init__(self, feature_data_path, subject_ids):
        #if not self.__class__.features:
        #    self.__class__.features = pickle.load(open(feature_data_path, 'rb'))
        self.video_features = pickle.load(open(feature_data_path, 'rb'))
        self.data = list()
        self.sequence_ids = list()
        for sub_ind in subject_ids:
            self.sequence_ids.extend(subject_ids[sub_ind])
        #print self.sequence_ids
        #exit()
        self.sequence_ids = np.random.permutation(self.sequence_ids)
        #print self.sequence_ids
        #for sequence_id, sequence_features in self.__class__.features.items():
        for sequence_id in self.sequence_ids:
            #print sequence_id
            #print self.sequence_ids
            #print len(self.video_features[sequence_id])
            self.data.append(self.video_features[sequence_id])
            #print len(sequence_features)
        self.max_node_label_len = np.max([len(metadata.subactivities), len(metadata.affordances)])

    def __getitem__(self, index):
        # index refers to a certain video
        edge_features = list()
        node_features = list()
        adj_mat = list()

        one_hot_relation_label = list()
        one_hot_graph_label = list()
        subject_index = list()

        one_hot_node_labels = list()
        #print len(self.data[index])
        #print type(self.data[index])

        for i in range(len(self.data[index])):
            #print self.data[index][i].keys()
            #exit()
            edge_features.append(self.data[index][i]['edge_features'])
            node_features.append(self.data[index][i]['node_features'])
            adj_mat.append(self.data[index][i]['adj_mat'])
            one_hot_relation_label.append(self.data[index][i]['relation_label'])
            one_hot_graph_label.append(self.data[index][i]['graph_label'])
            subject_index.append(self.data[index][i]['sub_ind'])
            node_label = self.data[index][i]['node_labels'].astype(np.int32)
            
            node_num = node_label.shape[0]
            one_hot_node_label = np.zeros((node_num, self.max_node_label_len))
            for v in range(node_num):
                one_hot_node_label[v, node_label[v]] = 1
            one_hot_node_labels.append(one_hot_node_label)

        edge_features = np.array(edge_features)
        node_features = np.array(node_features)
        adj_mat = np.array(adj_mat)
        one_hot_relation_label = np.array(one_hot_relation_label)
        one_hot_graph_label = np.array(one_hot_graph_label)
        subject_index = np.array(subject_index)
        one_hot_node_labels = np.array(one_hot_node_labels)
        #print one_hot_node_labels.shape
        #print adj_mat.shape
        #print one_hot_relation_label.shape


        #return np.transpose(edge_features, (2, 0, 1)), np.transpose(node_features, (1, 0)), adj_mat, one_hot_node_labels, one_hot_graph_label, self.sequence_ids[index]
        return edge_features,  node_features, adj_mat, one_hot_node_labels, one_hot_relation_label, one_hot_graph_label, self.sequence_ids[index], subject_index

    def __len__(self):
        return len(self.data)

