"""
Created on Jan 07, 2020

@author: Wang Ruiqi

Description of the file.
Changing the way of calculating MMD
predict subgraph using mse loss to constrain
stimulate full graph using mmd loss to constrain, using different scale full video representation
full video and partial video using different embedding parameters
no object feature
using fc to make temporal reasoning then using gcn to generate predicted part

version5.0

"""
import sys
import torch
import torch.nn as nn
import numpy as np
import gc
import random
import argparse
from torch.autograd import Variable
import torch.nn.functional as F
import math
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
import itertools
from torch.nn.parameter import Parameter
import pdb

class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))

class TemporalRelationGraph(torch.nn.Module):
    """
    LSTGN: Using multi-scale strategy to propagate imformation along edges
    """
    def __init__(self, opt):
        super(TemporalRelationGraph, self).__init__()
        
        #self.featdir = opt.featdir
        self.seq_size = opt.seq_size
        self.belta = opt.belta
        self.subsample_num = 10 # how many relations selected to sum up
        self.state_dim = opt.state_dim
        self.num_bottleneck = opt.num_bottleneck
        self.scales = [i for i in range(opt.num_frames, 0, -1)] # generate the multiple frame relations
        self.n_cluster = opt.n_cluster
        self.seq_size = opt.seq_size
        self.num_class = opt.classNum
        self.num_frames = opt.num_frames
        self.num_bottleneck = opt.num_bottleneck
        #self.d_pos = opt.d_pos
        self.cuda = opt.cuda
        self.hidden_size = opt.hidden_size
        self.rnn_num_layer = opt.rnn_num_layer
        self.rnn = nn.LSTM(self.num_bottleneck, self.hidden_size, self.rnn_num_layer, batch_first=True)
        self.classifier = nn.Sequential(
                        nn.Linear(self.hidden_size, self.state_dim),
                        nn.Tanh(),
                        nn.Linear(self.state_dim,self.num_class)
                        )
        # for action prediction, each step need select frames
        self.relations_scales = []
    
        self.subsample_scales = []

        if len(self.scales) == 0:
            self.scales = [1]
        for seq in range(1,self.seq_size+1):
            # 0~4, 0~5, ..., 0~14
            relations_scales_tmp = []
            subsample_scales_tmp = []
            for scale in self.scales:
                
                
                relations_scale = self.return_relationset(seq, min(scale,seq))
               
                relations_scales_tmp.append(relations_scale)
                subsample_scales_tmp.append(min(self.subsample_num, len(relations_scale)))
            self.relations_scales.append(relations_scales_tmp)
            self.subsample_scales.append(subsample_scales_tmp)# how many samples of relation to select in each forward pass

        '''
        self.attention =  nn.Sequential(
                        nn.Linear(((self.n_cluster+1)*self.state_dim), 1),
                        #nn.Linear(self.state_dim, 1),
                        nn.Sigmoid()
                        )
        '''
        self.fc_fusion_scales = nn.ModuleList() # high-tech modulelist
        for i in range(len(self.scales)):
            scale = self.scales[i]
            fc_fusion = nn.Sequential(
                        nn.ReLU(inplace=True),
                        nn.Linear(scale * ((self.n_cluster+1)*self.state_dim), self.num_bottleneck),
                        #nn.Linear(scale * self.state_dim, self.num_bottleneck),
                        #nn.ReLU(inplace=True),
                        #nn.Linear(self.num_bottleneck, self.num_bottleneck),#zhao added a linear layer in 6/24
                        #nn.ReLU(inplace=True),
                        #nn.Linear(self.num_bottleneck, self.num_bottleneck),#zhao added a linear layer in 6/25
                        #nn.ReLU(inplace=True),
                        #nn.Linear(self.num_bottleneck, self.num_bottleneck),#zhao added a linear layer in 6/27
                        #nn.ReLU(inplace=True),
                        #nn.Linear(self.num_bottleneck, self.num_bottleneck),#zhao added a linear layer in 6/28
                        nn.ReLU(inplace=True),
                        nn.Linear(self.num_bottleneck, self.num_class)
                        )
            self.fc_fusion_scales += [fc_fusion]
           
    
    def feat_mode(self):
        self.fc_fusion_scales_forfeat = copy.deepcopy(self.fc_fusion_scales)
        for i in range(len(self.scales)):
            self.fc_fusion_scales_forfeat[i] = self.fc_fusion_scales[i][:-1]

    def forward(self, graph_out, init_input, is_test):
        batch,_,_ = graph_out.size() #(batch, seq_size, state_dim)
        batch_variation = self.temporal_variation(init_input)  #object feature       
        batch_variation = batch_variation.view(batch, self.seq_size, -1) #(batch, seq_size, n_cluster*state_dim)
        
        if self.cuda:
            batch_variation = batch_variation.cuda()  
        new_input = torch.cat((batch_variation, graph_out), dim=2) #size=(batch, seq_size, (n_cluster+1)*state_dim)
        
        if self.cuda:
            result = torch.zeros((batch,self.seq_size,self.num_class)).cuda()
        else:
            result = torch.zeros((batch,self.seq_size,self.num_class))

        for seq_ind in range(1, self.seq_size+1):
            #*****adjust scales with the increasing input******#
            if is_test:
                new_scales = self.adjust_scale_test(seq_ind,self.num_frames*2,self.scales,self.subsample_num)
            else:
                new_scales = self.adjust_scale_train(seq_ind,self.num_frames*2,self.scales,self.subsample_num)
                #new_scales = self.scales
                
            if self.cuda:
                act_all = torch.zeros((batch,self.num_class)).cuda()
            else:
                act_all = torch.zeros((batch,self.num_class))

            new_input__ = new_input[:, :seq_ind, :]

            
            for scaleID in range(len(new_scales)):# for scale 1
                scale_index = self.scales.index(new_scales[scaleID])
                idx_relations_randomsample = self.select_idx(self.relations_scales[seq_ind-1][scale_index],new_scales[scaleID],seq_ind)
                
                for idx in idx_relations_randomsample: 
                    act_relation = new_input__[:, self.relations_scales[seq_ind-1][scale_index][idx], :] 
                    act_relation = act_relation.view(act_relation.size(0), -1)
                    act_relation = self.fc_fusion_scales[scale_index](act_relation)  #[batch,num_class]
                    act_all += act_relation
                   
            result[:, seq_ind-1, :] = act_all
            
        return result.view(batch*self.seq_size, -1)

    def temporal_variation(self, init_input): 
        '''
        Object features:Using VLAD to represent the temporal variation of each frame

        '''
        n_cluster = self.n_cluster
        cluster_inputs = init_input.view(-1, self.seq_size, init_input.size(1), init_input.size(-1))
        cluster_input = cluster_inputs.view(-1, self.seq_size*init_input.size(1), init_input.size(-1))
        # size = (batch, seq_size*n_nodes, state_dim)
        
        object_centers = []
        temporal_variation = []
        batch_variation = torch.zeros((cluster_inputs.size(0), self.seq_size, n_cluster, init_input.size(2)))
        for batch_ind in range(cluster_input.size(0)):
            
            cluster_input_ = cluster_input[batch_ind].cpu().detach().numpy()
            initial_centers = kmeans_plusplus_initializer(cluster_input_, n_cluster).initialize()
            kmeans_instance = kmeans(cluster_input_, initial_centers)
            kmeans_instance.process()
            centers = kmeans_instance.get_centers() # 'centers' is a list of ndarr
            ex = torch.zeros((cluster_input.size(1), len(centers))) #n_cluster =len(centers)
            for i in range(len(centers)):
                centers_i = np.tile(np.array(centers[i]), cluster_input.size(1)).reshape(cluster_input.size(1), -1)
                ex[:, i] = -self.belta*pow(np.linalg.norm((cluster_input_ - centers_i), ord=2), 2)   #||x-d_i||_2^2
          
            w = F.softmax(ex, dim=1)  #the normalized weight of descriptor x with respect to codeword d_i
            w = w.reshape(self.seq_size, -1, len(centers))  #(seq,n_node,n_cluster)
            x = cluster_input_.reshape(self.seq_size, -1, cluster_input.size(-1)) #(seq,n_node,nodde_state_dim)
        
            S_seq = []
            
            for seq_ind in range(self.seq_size):
                S = []
                # Object Feature: O_l
                for i in range(len(centers)): 
                    center_i = np.tile(np.array(centers[i]), cluster_inputs.size(2)).reshape(cluster_inputs.size(2), -1) #(n_node,node_state_dim)
                    w_i =  np.tile(w[seq_ind,:,i].reshape(-1,1), cluster_inputs.size(-1))
                    # wi: (n_nodes, state_dim), weight corresponding to cluster_i
                    center_varaiation = np.sum(np.multiply(w_i, (x[seq_ind,:,:]-center_i)), axis=0)
                    S.append(center_varaiation)
                S_seq.append(np.array(S))
            batch_variation[batch_ind, :, :len(centers), :] = torch.from_numpy(np.array(S_seq))  # (batch, seq, n_cluster, state_dim)
           
        return batch_variation


    def return_relationset(self, num_frames, num_frames_relation):  #num_frames_relation is scale
        '''select video frame in order'''
        return list(itertools.combinations([i for i in range(num_frames)], num_frames_relation))

    def select_idx(self, relations_scales, scale, num_frames):
        def is_Arithmetic_progression(arr,distance):
            arr_dis= np.diff(arr)
            if (arr_dis == distance).all():
                return True
            else:
                return False
        idx = list()
        idx_count=0
        division = math.floor(num_frames/scale)
        for relations_scale in relations_scales:
            if is_Arithmetic_progression(relations_scale,division):
                idx.append(idx_count)
            idx_count += 1
        return idx


    def return_relationset_random(self, num_frames, num_frames_relation):  
        '''select video frame in order'''
        import itertools
        permutations=list(itertools.combinations([i for i in range(num_frames)], num_frames_relation))
        new_permutations=list()
        for i in range(len(permutations)):
            tmp = list(permutations[i])
            random.shuffle(tmp)
            tmp = tuple(tmp)
            new_permutations.append(tmp)
      
        return new_permutations

    def adjust_scale_train(self,observe_ratio,scale_max,scales,subsample_num):
        '''adjust scale selection with the increasing video length '''
        if observe_ratio<scale_max:
            return scales[-min(subsample_num,observe_ratio):]
        else:
            scale_min_num = min(subsample_num,observe_ratio)
            new_scales = scales[:scale_min_num+1]
            return new_scales

    def adjust_scale_test(self,num_frames,scale_max,scales,subsample_num):
        '''adjust scale selection with the increasing video length '''
        if num_frames<scale_max:
            return scales[-num_frames:]
        else:
            scale_min_num = min(subsample_num,num_frames)
            new_scales = scales[:scale_min_num+1]
            return new_scales

class Residual_Block(nn.Module):
    def __init__(self, state_dim,node_state_dim,edge_state_dim,tem_feat_dim):
        super(Residual_Block, self).__init__()
        self.node_state_dim = node_state_dim
        self.edge_feat_dim = edge_state_dim
        self.state_dim = state_dim
        self.tem_feat_dim = tem_feat_dim
        #self.linear_node = nn.Linear(self.node_state_dim, self.state_dim )
        #self.linear_edge = nn.Linear(self.edge_feat_dim, self.state_dim )
        self.relu = nn.ReLU(inplace=True)
        self.linear = nn.Linear(state_dim+tem_feat_dim, state_dim)
    #def forward(self, graph_out, init_input,edge_state): 
    def forward(self, graph_out,init_input,edge_state, tem_feature):
    
        feat_input = torch.cat((init_input,edge_state),1)
        residual = torch.mean(feat_input, dim=1)
        
        out = graph_out+residual
        out = self.relu(out)
        
        return out            

class Propogator(nn.Module):
    """
    Gated Propogator for GGNN
    Using LSTM gating mechanism
    """
    def __init__(self, state_dim):
        super(Propogator, self).__init__()

        self.reset_gate = nn.Sequential(
            nn.Linear(state_dim*2, state_dim),
            nn.Sigmoid()  #TODO maybe it should add BN layer if feature change in the future
        )
        self.update_gate = nn.Sequential(
            nn.Linear(state_dim*2, state_dim),
            nn.Sigmoid()
        )
        self.tansform = nn.Sequential(
            nn.Linear(state_dim*2, state_dim),
            nn.Tanh()
        )

    def forward(self, state_edge, state_cur, A):
        #print A.size()
        a_cur = torch.bmm(A, state_edge)  #[n_node,state_dim]
        #print(a_cur)
        #print ("a_cur:", a_cur.size())
        a = torch.cat((a_cur, state_cur), 2)
        #print ("a:", a.size())
        r = self.reset_gate(a)
        z = self.update_gate(a)
        joined_input = torch.cat((a_cur, r * state_cur), 2)
        h_hat = self.tansform(joined_input)

        output = (1 - z) * state_cur + z * h_hat  #  [batch_size, n_node, state_dim]

        return output

class SpatialRelation(nn.Module):
    """
    modified from GGNN, add external edge states
    Mode: Graph-level output
    Implementation based on https://arxiv.org/abs/1511.05493
    """
    def __init__(self, opt):
        super(SpatialRelation, self).__init__()
        #assert (opt.state_dim >= opt.annotation_dim,  \
        #        'state_dim must be no less than annotation_dim')
        self.state_dim = opt.state_dim
        self.node_feat_dim = opt.node_feat_dim
        self.edge_feat_dim = opt.edge_feat_dim
        self.tem_feat_dim = opt.tem_feat_dim
        #self.n_node = opt.n_node
        self.n_steps = opt.n_steps

        #for i in range(self.n_edge_types):
            # edge embedding--undirected edge

        #    link_fc = nn.Linear(self.edge_feat_dim, self.state_dim)
        #    self.add_module("link_{}".format(i), link_fc)

        #self.link_fcs = AttrProxy(self, "link_")
        self.link_fc = nn.Linear(self.state_dim, self.state_dim)

        # Propogation Model
        self.propogator = Propogator(self.state_dim)
        self.residual_block = Residual_Block(self.state_dim, self.node_feat_dim, self.edge_feat_dim,self.tem_feat_dim)

        # Output Model  Using soft attention mechanism to decide which nodes are relevant to the task
        self.attention = nn.Sequential(
            nn.Linear(self.state_dim, self.state_dim),
            nn.Tanh(),
            nn.Linear(self.state_dim, 1)
        )

        self.out = nn.Sequential(
            nn.Linear(self.state_dim, self.state_dim),
            nn.Tanh(),
            #nn.Linear(self.state_dim, 1)
        )

        self.result = nn.Tanh()

        self.linear_node = nn.Linear(self.node_feat_dim, self.state_dim )
        self.linear_edge = nn.Linear(self.edge_feat_dim, self.state_dim )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, prop_state, edge_states, A,tem_features):  # prop_State:[batch*seq_size,n_nodes,state_dim]
        
        init_node_states = prop_state
        init_edge_states = edge_states

        for i_step in range(self.n_steps):
         
            #update_state = self.link_fc(edge_states) ##embedding updated node states
            edge_states = self.link_fc(edge_states) ##embedding updated node states
            prop_state = self.propogator(edge_states, prop_state, A)

        #join_state = torch.cat((prop_state, annotation), 2)
        atten = self.attention(prop_state)  # [batch_size, n_node, 1]
        A = torch.transpose(atten, 2, 1)  # [batch,1,n_node]
        A = F.softmax(A,dim = 2) 
        out = self.out(prop_state)
        mul = torch.bmm(A,out)
        #output = output.sum(2)
        w_sum = torch.squeeze(mul)
        graph_out = self.result(w_sum)
        #res = self.residual_block(graph_out, init_node_states, init_edge_states)  
        res = self.residual_block(graph_out, init_node_states, init_edge_states,tem_features)

        return res, mul, edge_states

class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.seq_size = opt.seq_size
        self.hidden_size = opt.hidden_size
        self.n_class = opt.classNum
        self.n_rclass = opt.rclassNum
        self.n_node_class = opt.nodeclassNum
        self.rnn_num_layer = opt.rnn_num_layer
        self.state_dim = opt.state_dim
        self.num_bottleneck = opt.num_bottleneck
        self.node_feat_dim = opt.node_feat_dim
        self.edge_feat_dim = opt.edge_feat_dim
        self.num_layers = opt.rnn_num_layer
        self.tem_feat_dim = opt.tem_feat_dim
        self.spatial_relation = SpatialRelation(opt)
        #self.spatial_relation = newSemantic(opt)
        self.temporal_relation = TemporalRelationGraph(opt)
        self.residual_block = Residual_Block(self.state_dim, self.node_feat_dim, self.edge_feat_dim,self.tem_feat_dim)
        
        self.linear = nn.Sequential(
                        nn.Linear(self.state_dim+self.tem_feat_dim, self.state_dim),
                        nn.Tanh(),
                        )
     
        self.linear_node = nn.Sequential(
            nn.Linear(self.node_feat_dim, self.state_dim),
            nn.ReLU()
        )
        self.linear_edge = nn.Sequential(
            nn.Linear(self.edge_feat_dim, self.state_dim),
            nn.ReLU(inplace=True)
        )
        
        self._initialization()

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def feat_mode(self):
        self.temporal_relation.feat_mode()

    def load_state_dict_(self,target_weights):
    #f = open('./tensor_name.txt','w')
    #f1 = open('./tensor_name_s3d.txt','w')
    #if not os.path.isfile('./tensor_name.txt'):
        #print("-------")
        #os.wknod('./tensor_name.txt')
        #os.wknod('./tensor_name_s3d.txt')
                
        target_weights = target_weights['state_dict']
        #print(set(target_weights))
        own_state=self.state_dict()    
            
        for name, param in target_weights.items():#the parameter in trained model
            print(name)
            if name in own_state:
                if isinstance(param,nn.Parameter): 
                    param=param.data
                try:
                    if len(param.size())==5 and param.size()[3] in [3,7]:
                        own_state[name][:,:,0,:,:]=torch.mean(param,2)
                    else:
                        own_state[name].copy_(param)
                except Exception:
                    raise RuntimeError('While copying the parameter named {}.\
                                           whose dimensions in the model are {} and \
                                           whose dimensions in the checkpoint are {}.\
                                           '.format(name,own_state[name].size(),param.size()))
            else:
                print ('{} meets error in locating parameters'.format(name))
        missing=set(own_state.keys())-set(target_weights.keys())
        #f.close()
        print (len(set(own_state.keys())),len(set(target_weights.keys())))
        print ('{} keys are not holded in target checkpoints'.format(len(missing)))
    
    def forward(self, init_input, edge_states, A, true_lengths, tem_features, is_test): # init_input:[batch*seq,n_nodes,state_dim]
        batch, _, _ = init_input.size()
      
        init_input_ = self.linear_node(init_input)
        edge_states_ = self.linear_edge(edge_states)
        without_spatial = False
        if without_spatial:   #zhao added it in 7.4
            graph_output, nodes_out, edges_out, Attention_value = torch.sum(init_input_, dim=1).view(-1, self.state_dim), init_input_, edge_states_, 1
            graph_output = graph_output.contiguous().view(-1, self.seq_size, self.state_dim)  # [batch,seq,n_nodes]
            vis_loss = self.spatial_relation(nodes_out, edges_out, A, true_lengths)
            graph_outputs = graph_output.view(batch, -1) 
            output = graph_outputs.view(-1, self.seq_size, self.state_dim)
            #tempora_input=torch.cat((output,tem_features),2)
            output = self.temporal_relation(output, init_input_, is_test)
        else:
            graph_output, nodes_out, edges_out = self.spatial_relation(init_input_, edge_states_, A, tem_features)# return attention
            # Lin: graph_output.shape = (batch*seq, state_dim)
            graph_output = graph_output.contiguous().view(-1, self.seq_size, self.state_dim)  # [batch,seq,n_nodes]
            #vis_loss = self.spatial_relation(nodes_out, edges_out, A, true_lengths)
            graph_outputs = graph_output.view(batch, -1)  
            output = graph_outputs.view(-1, self.seq_size, self.state_dim)
            # Tempo relation Net reads 'graph_output' and 'node_feat_input(dim-256)'
            temporal_input = torch.cat((output,tem_features),2)
            temporal_input_ = self.linear(temporal_input)
            
            #print(output.size())
            output = self.temporal_relation(temporal_input_, init_input_, is_test)
       
        return output



'''
if __name__ == "__main__":
    batch_size = 24
    seq_size = 10
    num_frames = 5
    num_class = 10
    img_feature_dim = 256
    parser = argparse.ArgumentParser()
   
    parser.add_argument('--batch_size', type=int, default=24, help='input batch size')
    parser.add_argument('--hidden_size', type=int, default=128, help='hidden size of rnn')
    parser.add_argument('--seq_size', type=int, default=10, help='sequence length of rnn')
    parser.add_argument('--rnn_num_layer', type=int, default=1, help='layer number of rnn')
    parser.add_argument('--classNum', type=int, default=13, help='number of classes')
    parser.add_argument('--rclassNum', type=int, default=10, help='number of relation classes')
    parser.add_argument('--nodeclassNum', type=int, default=12, help='number of classes')
    #parser.add_argument('--n_nodes', type=int, default=3, help='number of nodes in graph')
    parser.add_argument('--n_type_node', type=int, default=6, help='#objects+human')
    parser.add_argument('--node_feat_dim', type=int, default=250, help='node state size')
    parser.add_argument('--edge_feat_dim', type=int, default=250, help='edge state size')
    parser.add_argument('--state_dim', type=int, default=256, help='dim of annotation')
    parser.add_argument('--num_bottleneck', type=int, default=56, help='dim of temporal reasoning module')
    parser.add_argument('--num_frames', type=int, default=4, help='number of sampled frames in each segment ')
    parser.add_argument('--n_steps', type=int, default=1, help='propogation steps number of GGNN')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    opt = parser.parse_args()
    opt.cuda = True
    input_var = Variable(torch.randn(opt.batch_size,opt.seq_size, opt.state_dim))
    node_input = Variable(torch.randn(opt.batch_size*opt.seq_size,opt.n_type_node,opt.node_feat_dim))
    edge_input = Variable(torch.randn(opt.batch_size*opt.seq_size,  opt.n_type_node*opt.n_type_node,opt.edge_feat_dim))
    adj_mat = Variable(torch.randn(opt.batch_size*opt.seq_size, opt.n_type_node, opt.n_type_node*opt.n_type_node))
    true_lengths =  Variable(torch.randn(opt.batch_size,1))
    model = Model(opt)

    output,vis_loss = model(node_input,edge_input,adj_mat,true_lengths)
    print(output)

'''