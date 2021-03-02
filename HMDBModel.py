"""
Created on Jan 07, 2020

@author: Wang Ruiqi

Description of the file.
Adding new part to version 4.0 of work "Spatial Temporal Relation Reasoning for Action Prediction in Videos"

version4.0

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


## TODO- Check if work
class Semantic(nn.Module):
    
    #Semantic Transformation Mmodule  for CAD120
    #Implementation based on https://arxiv.org/abs/1811.10696
    
    def __init__(self, opt):
        super(Semantic, self).__init__()
        self.loss=0.
     
        self.state_dim = opt.state_dim
        self.seq_size = opt.seq_size
        self.projection_layer1 = nn.Linear(self.state_dim, 300)
        self.projection_layer2 = nn.Linear(self.state_dim, 300)
        self.projection_layer3 = nn.Linear(self.state_dim, 300)
        self.embedding = nn.Sequential(
            nn.Linear(900, 500),
            nn.ReLU(inplace=True)
        )
        self.vis_loss = nn.MSELoss()
    def forward(self, nodes_feats, edges_feats, adj_mat, true_lengths):
        # Lin added 'torch.set_grad_enabled' on Sept. 3rd
        #with torch.set_grad_enabled(False):
            #vis_trans = []
            
        losses = 0.
        
        batseq, node_num, node_feat_dim =nodes_feats.size()
        _, edge_num, edge_feat_dim =edges_feats.size()
        #nodes_feats = nodes_feats.view(-1, self.seq_size, node_num, node_feat_dim)
        #edges_feats = edges_feats.view(-1, self.seq_size, edge_num, edge_feat_dim)
        adj_mat = adj_mat.view(-1, self.seq_size, node_num, edge_num)
        batch, _, _, _ = adj_mat.size()
        for i in range(node_num):
            for j in range(node_num):
                obj1_feat = nodes_feats[:,i,:]
                f_i = self.projection_layer1(obj1_feat)
                rela_feat = edges_feats[:,i*node_num+j, :]
                f_ij = self.projection_layer2(rela_feat)
                obj2_feat = nodes_feats[:, j, :]
                f_j =self.projection_layer3(obj2_feat)


                losses += self.vis_loss(f_j,(f_ij+f_i)).item()
                #pdb.set_trace()


        losses = losses / batch
        return losses



class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        #support = torch.mm(input, self.weight)  ##changed by wrq2020/01/14 to adapt to batch data
        support = torch.matmul(input, self.weight)  #(batchsize,node_num,out_features)
        #output = torch.spmm(adj, support)  ##changed as below
        output = torch.bmm(adj, support)  #sparse matrix multiplication has been changed to common matrix multiplication
        output = self.act(output) #(batchsize,node_num,out_features)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'        

class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        #adj = self.act(torch.mm(z, z.t())) #changed by wrq2020/01/14 to adapt to batch data
       
        adj = self.act(torch.bmm(z, z.permute(0,2,1)))
        return adj

class GCNModelVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelVAE, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def encode(self, x, adj):
        
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
           
        else:
            return mu

    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)  #rebuilt adjacency matrix
        return self.dc(z), mu, logvar

class GenerateGraphRep(torch.nn.Module):     
    """generate graph-level representation"""
    def __init__(self, opt,input_dim):
        super(GenerateGraphRep, self).__init__()   
        self.state_dim = opt.state_dim

        self.attention = nn.Sequential(
            nn.Linear(input_dim, self.state_dim),
            nn.Tanh(),
            nn.Linear(self.state_dim, 1)
        )

        self.out = nn.Sequential(
            nn.Linear(input_dim, self.state_dim),
            nn.Tanh(),
            #nn.Linear(self.state_dim, 1)
        )

        self.result = nn.Sigmoid()

    def forward(self,node_states):

        atten = self.attention(node_states)  # [batch_size, n_node, 1]
        A = torch.transpose(atten, 2, 1)  # [batch,1,n_node]
        A = F.softmax(A,dim = 2) 
        out = self.out(node_states)
        mul = torch.bmm(A,out)
        w_sum = torch.squeeze(mul)
        graph_out = self.result(w_sum)

        return graph_out

class PredictGraph(torch.nn.Module):
    """predict subgraph representation"""
    def __init__(self, opt):
        super(PredictGraph, self).__init__()
        self.state_dim = opt.state_dim
        self.tem_feat_dim = opt.embedding_dim
        self.cuda = opt.cuda
        self.dropout = opt.dropout
        self.predict_node_num = opt.predict_node_num
        self.n_cluster = opt.n_cluster
        input_dim=self.state_dim+self.tem_feat_dim
        self.GVAE=GCNModelVAE(input_dim,opt.gcn_dim1,opt.gcn_dim2,opt.dropout)
        self.gc = GraphConvolution(input_dim, opt.gcn_dim1,  self.dropout, act=F.relu)
        self.subgraph_rep = GenerateGraphRep(opt,self.state_dim)
        self.fullgraph_rep = GenerateGraphRep(opt,input_dim+self.state_dim)
        
    def predic_node(self,current_nodes,rebuilt_matrix):
        '''generate node representation based on the new adjancey matrix'''
        predict_node =  self.gc(current_nodes,rebuilt_matrix)

        return predict_node

    def forward(self, current_nodes):
        batch,node_num,node_dim = current_nodes.size()
        if self.cuda:
            adj = torch.ones((batch,node_num,node_num)).cuda()
        else:
            adj=  torch.ones((batch,node_num,node_num))
        
        #recovered_A, mu, logvar = self.GVAE(current_nodes, adj)
        
        '''generate predicted sub-graph representation'''
        
        predict_node = self.predic_node(current_nodes,adj)
        predict_nodes = predict_node  
        predict_subgraph=self.subgraph_rep(predict_nodes)


        '''generate simulated full graph'''
       
        simulate_fullgraph =  self.fullgraph_rep(torch.cat((current_nodes, predict_nodes), dim=2))

        return predict_subgraph,simulate_fullgraph






class TemporalRelationGraph(torch.nn.Module):
    def __init__(self, opt):
        super(TemporalRelationGraph, self).__init__()
        self.seq_size = opt.seq_size
        self.belta = opt.belta
        self.subsample_num = 10 # how many relations selected to sum up
        self.state_dim = opt.state_dim
        self.num_bottleneck = opt.num_bottleneck
        #self.scales = [i for i in range(opt.num_frames, 1, -1)] # generate the multiple frame relations
        self.scales = [i for i in range(opt.num_frames, 0, -1)]
        self.seq_size = opt.seq_size
        self.num_class = opt.classNum
        self.num_frames = opt.num_frames
        self.num_bottleneck = opt.num_bottleneck
        self.cuda = opt.cuda
        self.n_cluster = opt.n_cluster  # clustering number of object representation 
        self.tem_feat_dim = opt.tem_feat_dim
        self.hidden_size = opt.hidden_size
        self.rnn_num_layer = opt.rnn_num_layer
        self.project_dim = opt.project_dim
        self.rnn = nn.LSTM(self.num_bottleneck, self.hidden_size, self.rnn_num_layer, batch_first=True)
        self.classifier = nn.Sequential(
                        nn.Linear(self.state_dim, self.state_dim),
                        nn.ReLU(inplace=True),
                        nn.Linear(self.state_dim,self.num_class)
                        )
        self.linear = nn.Sequential(
                        nn.Linear(self.state_dim+self.tem_feat_dim, self.state_dim),
                        nn.Tanh(),
                        )
        self.Linear = nn.Sequential(
                        nn.Linear(self.state_dim, self.num_bottleneck),
                        nn.ReLU(inplace=True),
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
                ##relations_scale = self.return_relationset(self.num_frames+seq, scale)
                relations_scale = self.return_relationset(seq, min(scale,seq))
                relations_scales_tmp.append(relations_scale)
                subsample_scales_tmp.append(min(self.subsample_num, len(relations_scale)))
            self.relations_scales.append(relations_scales_tmp)
            self.subsample_scales.append(subsample_scales_tmp)# how many samples of relation to select in each forward pass
        
        self.graph_rep_part = GenerateGraphRep(opt,self.state_dim+self.tem_feat_dim)
        self.graph_rep_full = GenerateGraphRep(opt, self.state_dim+self.tem_feat_dim)
        self.PredictGraph = PredictGraph(opt)
        self.graph_loss = nn.MSELoss()   
        self.bn = nn.BatchNorm1d(self.state_dim,affine=True)
        
          


    def forward(self, graph_out, init_input, is_test):
        batch, _, _ = graph_out.size()
        MSE_losses = 0.
        MMD_losses = 0.

    
        true_graph_rep =  self.graph_rep_full(graph_out)  # true graph representation based on the complete video
        # size=(batch, seq_size, (n_cluster+1)*state_dim)
       
        if self.cuda:
            result = torch.zeros((batch,self.seq_size,self.num_class)).cuda()
        else:
            result = torch.zeros((batch,self.seq_size,self.num_class))

        ##new_input_con = temporalvar[:, :1, :].repeat(1, self.num_frames, 1)
        # Padding
        # size=(batch, num_frames, 1)
        ##new_input = torch.cat((new_input_con, temporalvar), dim=1) # (batch, num_frames+seq_size, (n_cluster+1)*state_dim) # (batch, num_frames+seq_size, (n_cluster+1)*state_dim + d_pos)

        '''TODO:change sampling strategy to adapt to scale-1'''
        one_node = self.linear(new_input[:, 0, :])
        result[:, 0, :] = self.classifier(one_node)        
        
        for seq_ind in range(1,self.seq_size+1):   
            
            #*****adjust scales with the increasing input******#
            if is_test:
                new_scales = self.adjust_scale_test(seq_ind,self.num_frames,self.scales)
            else:
                new_scales = self.adjust_scale_train(seq_ind,self.num_frames,self.scales)
                
            if self.cuda:
                result_all = torch.zeros((batch,self.num_class)).cuda()
            else:
                result_all = torch.zeros((batch,self.num_class))

            if self.cuda:
                predict_subgraphs = torch.zeros((batch,self.num_bottleneck)).cuda()
                predict_fullgraph = torch.zeros((batch,self.state_dim)).cuda()
            else:
                predict_subgraphs = torch.zeros((batch,self.num_bottleneck))
                predict_fullgraph = torch.zeros((batch,self.state_dim))

            observed_part = new_input[:, :seq_ind, :]

            
            if  self.training: 
                unobserved_part = new_input[:, seq_ind:, :]
                if seq_ind< self.seq_size:
                    true_un_graph_rep =  self.graph_rep_part(unobserved_part)  # true graph representation based on the unobserved video


            for scaleID in range(len(new_scales)):
                
                scale_index = self.scales.index(new_scales[scaleID])
         
                idx_relations_randomsample = self.select_idx(self.relations_scales[seq_ind-1][scale_index],new_scales[scaleID],seq_ind)
                # random sample subsample_scales[seq_ind-1][scaleID] frames, in relations_scales[seq_ind-1][scaleID]
                
                for idx in idx_relations_randomsample:

                    ob_act_relation = observed_part[:, self.relations_scales[seq_ind-1][scale_index][idx], :]
                    
                    predict_subgraph, simulated_fullgraph = self.PredictGraph(ob_act_relation)

                    if  self.training and seq_ind< (self.seq_size+1): 
                    #MSE_losses.append(self.graph_loss(predict_subgraph,true_un_graph_rep).item())

                        predict_subgraph = self.bn(predict_subgraph)
                        true_un_graph_rep = self.bn(true_un_graph_rep)
                        predict_subgraph = self.Linear(predict_subgraph)
                        gt_un_graph_rep = self.Linear(true_un_graph_rep)
                        MSE_losses += self.graph_loss(predict_subgraph,gt_un_graph_rep).item()
                     
                        #print(predict_subgraph)
                        #print(gt_un_graph_rep)
                        #print("mse loss: %.4f"%self.graph_loss(predict_subgraph,gt_un_graph_rep).item())
                        #exit()

                    result_all += self.classifier(simulated_fullgraph)
                    #predict_subgraphs += predict_subgraph
                    predict_fullgraph += simulated_fullgraph

                    
                    

            result[:, seq_ind-1, :] = result_all
            #MMD_losses.append(self.graph_loss(predict_fullgraph/len(new_scales),true_graph_rep).item())
            if  self.training: 
                avg_predict = self.bn(predict_fullgraph/len(new_scales))
                true_graph_rep = self.bn(true_graph_rep)
                #print(avg_predict)
                #print(true_graph_rep)                
                avg_predict = self.Linear(avg_predict)
                
                gt_full_graph_rep = self.Linear(true_graph_rep)

                MMD_losses += self.graph_loss(avg_predict,gt_full_graph_rep).item()
                #print(avg_predict)
                #print(gt_full_graph_rep)
                #print("mmd loss:%.4f"%self.graph_loss(avg_predict,gt_full_graph_rep).item())
               # exit()

        #print(MSE_losses)
        #print(MMD_losses)
        if  self.training: 
            mse_loss = MSE_losses/(self.seq_size-1)
            mmd_loss = MMD_losses/(self.seq_size-1)
           
        else:
            mse_loss = 0.
            mmd_loss = 0.
            #print("this is testing loss")
        return result.view(batch*self.seq_size, -1),mse_loss,mmd_loss 

    def temporal_variation(self, init_input):
        #using VLAD to represent the temporal variation of each frame
        #batch,_,_ = init_input.size()  
        
        n_cluster = self.n_cluster
        cluster_inputs = init_input.view(-1, self.seq_size, init_input.size(1), init_input.size(-1))
        cluster_input  = cluster_inputs.view(-1, self.seq_size*init_input.size(1), init_input.size(-1))
        # size = (batch, seq_size*n_nodes, state_dim)

        object_centers  = []
        temporal_variation = []
        batch_variation = torch.zeros((cluster_inputs.size(0), self.seq_size, n_cluster, init_input.size(2)))
        for batch_ind in range(cluster_input.size(0)):
            # to cpu, again??
            cluster_input_  = cluster_input[batch_ind].cpu().detach().numpy()
            initial_centers = kmeans_plusplus_initializer(cluster_input_, n_cluster).initialize()
            kmeans_instance = kmeans(cluster_input_, initial_centers)
            kmeans_instance.process()
            #clusters = kmeans_instance.get_clusters()
            # 'centers' is a list of ndarr
            centers = kmeans_instance.get_centers()
            # Lin commented on Sept. 3rd
            #n_cluster =len(centers)
            #assert len(centers)==n_cluster, '{} centers'.format(len(centers))
            ex = torch.zeros((cluster_input.size(1), len(centers)))
            for i in range(len(centers)):
                centers_i = np.tile(np.array(centers[i]), cluster_input.size(1)).reshape(cluster_input.size(1), -1)
               # print np.array(centers[i]).reshape(1,cluster_input.size(2)).shape
                ex[:, i]  = -self.belta*pow(np.linalg.norm((cluster_input_ - centers_i), ord=2), 2)   #||x-d_i||_2^2
            #print clusters
            #object_centers.append(centers)
            w = F.softmax(ex, dim=1)  #the normalized weight of descriptor x with respect to codeword d_i
            w = w.reshape(self.seq_size, -1, len(centers))  #(seq,n_node,n_cluster)
            x = cluster_input_.reshape(self.seq_size, -1, cluster_input.size(-1)) #(seq,n_node,nodde_state_dim)
        
            S_seq = []
            # Video O_l
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
            batch_variation[batch_ind, :, :len(centers), :] = torch.from_numpy(np.array(S_seq))
            #               (batch,   seq, n_cluster, state_dim)
        
        return batch_variation

    def return_relationset(self, num_frames, num_frames_relation):  
        '''select video frame in order'''
        return list(itertools.combinations([i for i in range(num_frames)], num_frames_relation))

    def return_relationset_random(self, num_frames, num_frames_relation):  
        '''select video frame in order'''
        permutations=list(itertools.combinations([i for i in range(num_frames)], num_frames_relation))
        new_permutations=list()
        #print type(permutations[0])
        for i in range(len(permutations)):
            tmp = list(permutations[i])
            random.shuffle(tmp)
            tmp = tuple(tmp)
            new_permutations.append(tmp)
      
        return new_permutations

    def select_idx(self,relations_scales,scale,num_frames):
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

    def adjust_scale_train(self,observe_ratio,scale_max,scales):
        '''adjust scale selection with the increasing video length '''
        if observe_ratio<scale_max:
            return scales[-min(5,observe_ratio):]
        else:
            scale_min_num = min(5,observe_ratio)
            new_scales = scales[:scale_min_num+1]
            return new_scales

    def adjust_scale_test(self,num_frames,scale_max,scales):
        '''adjust scale selection with the increasing video length '''
        if num_frames<scale_max:
            return scales[-num_frames:]
        else:
            scale_min_num = min(5,num_frames)
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
        self.embedding_dim = opt.embedding_dim
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
        self.residual_block = Residual_Block(self.state_dim, self.node_feat_dim, self.edge_feat_dim,self.embedding_dim)

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
        
        self.node_feat_dim = opt.node_feat_dim
        self.edge_feat_dim = opt.edge_feat_dim
        self.num_layers = opt.rnn_num_layer
        self.tem_feat_dim = opt.tem_feat_dim
        self.embedding_dim = opt.embedding_dim
        self.spatial_relation = SpatialRelation(opt)
        self.semantic_relation = Semantic(opt)
        self.temporal_relation = TemporalRelationGraph(opt)

        #self.avg_pooling = nn.AvgPool1d(3,stride=1)
        self.linear_node = nn.Sequential(
            nn.Linear(self.node_feat_dim, self.state_dim),
            nn.ReLU(inplace=True)
        )
        self.linear_edge = nn.Sequential(
            nn.Linear(self.edge_feat_dim, self.state_dim),
            nn.ReLU(inplace=True)
        )
        self.linear_embedding = nn.Sequential(
            nn.Linear(self.tem_feat_dim, self.embedding_dim),
            nn.ReLU(inplace=True)
        )
        
        self.classifier =nn.Sequential (
            nn.Linear(self.hidden_size, self.state_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.state_dim, self.n_class),

        )
        
        self._initialization()

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, init_input, edge_states, A, true_lengths, tem_features,is_test): # init_input:[batch*seq,n_nodes,state_dim]
        batch, _, _ = init_input.size()
        
        init_input_ = self.linear_node(init_input)
        edge_states_ = self.linear_edge(edge_states)
        tem_features = self.linear_temp(tem_features)
        #print "Node and edge have been tranfered to the same size"
        without_spatial = False
        if without_spatial:
            graph_output, nodes_out, edges_out = torch.sum(init_input_, dim=1).view(-1, self.state_dim), init_input_, edge_states_
            graph_output = graph_output.contiguous().view(-1, self.seq_size, self.state_dim)  # [batch,seq,n_nodes]
            ##vis_loss = self.semantic_relation(nodes_out, edges_out, A, true_lengths)
            #print "Visual semantic loss calculation finished."
            graph_outputs = graph_output.view(batch, -1)  # if residual before lstm it should be done
            
            output = graph_outputs.view(-1, self.seq_size, self.state_dim)
        else:

            #graph_output, nodes_out, edges_out = self.graph_net(init_input_, edge_states_, A)
            graph_output, nodes_out, edges_out = self.spatial_relation(init_input_, edge_states_, A,tem_features)
            # Lin: graph_output.shape = (batch*seq, state_dim)
            #print "Spatial relation reasoning finished."
            graph_output = graph_output.contiguous().view(-1, self.seq_size, self.state_dim)  # [batch,seq,n_nodes]
            ##vis_loss = self.semantic_relation(nodes_out, edges_out, A, true_lengths)
            #print "Visual semantic loss calculation finished."
            graph_outputs = graph_output.view(batch, -1)  # if residual before lstm it should be done
            

            output = graph_outputs.view(-1, self.seq_size, self.state_dim)

            # Tempo relation Net reads 'graph_output' and 'node_feat_input(dim-256)'
        tempora_input=torch.cat((output,tem_features),2)

        #output = self.temporal_relation(output, init_input_, is_test)
        output,mse_loss,mmd_loss = self.temporal_relation(tempora_input, init_input_, is_test)

       
        return output,mse_loss,mmd_loss##, relation_pre_out#, self.classfierNode(nodes_out)



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