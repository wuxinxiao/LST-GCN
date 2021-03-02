"""
Created on Mar 07, 2019

@author: Wang Ruiqi

Description of the file.


version3.0

"""
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import time
import numpy as np
import pdb
import torch.nn as nn
#import torchvision.transforms as transforms
'''Lin commented on Sept. 1st
import pretrainedmodels
import pretrainedmodels.utils as utils
'''
import sklearn.metrics

#from tensorboardX import SummaryWriter



def evaluation(pred_node_labels, node_labels):
    '''
    same as in GPNN
    :param pred_node_labels:
    :param node_labels: one hot label for nodes
    :return:
    '''
    np_pred_node_labels = pred_node_labels.data.cpu().numpy()
    np_node_labels = node_labels.data.cpu().numpy()
    #print np_pred_node_labels.shape
    #print np_node_labels.shape

    predictions = list()
    ground_truth = list()

    error_count = 0
    total_nodes = 0
    for batch_i in range(np_pred_node_labels.shape[0]):
        total_nodes += np_pred_node_labels.shape[0]
        pred_indices = np.argmax(np_pred_node_labels[batch_i, :,:],1)
        indices = np.argmax(np_node_labels[batch_i,  :,:], 1)
        predictions.extend(pred_indices)
        ground_truth.extend(indices)

        for node_i in range(np_pred_node_labels.shape[1]):
            if pred_indices[node_i] != indices[node_i]:
                error_count += 1
    #print len(predictions)
    #print len(ground_truth)
    #print predictions[0].shape
    #print ground_truth[0].shape
    #exit()
    return error_count/float(total_nodes), total_nodes, predictions, ground_truth

def masked_cross_entropy(logits, target, length):

    def sequence_mask(sequence_length, max_len=None):
        if max_len is None:
            max_len = sequence_length.data.max()
        batch_size = sequence_length.size(0)
        # seq_range = torch.range(0, max_len - 1).long()
        seq_range = torch.arange(0, max_len).long()
        seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
        seq_range_expand = Variable(seq_range_expand)
        if sequence_length.is_cuda:
            seq_range_expand = seq_range_expand.cuda()
        seq_length_expand = (sequence_length.unsqueeze(1)
                             .expand_as(seq_range_expand))
        #print type(seq_length_expand)
        #print type(seq_range_expand)
        return seq_range_expand < seq_length_expand


    length = Variable(length)
    if torch.cuda.is_available():
        length = length.cuda()

    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
    """

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = F.log_softmax(logits_flat, dim=1)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()
    return loss


def train(epoch, dataloader, model, criterion1,criterion2, optimizer, opt, writer, log_file):
    model.train()


    train_loss = 0.
    vis_losses = 0.
    mse_losses = 0.
    mmd_losses = 0.
    test_out=[]
    test_out_relation = []
    train_correct = np.zeros([1,opt.seq_size])

    correct = np.zeros([1,opt.seq_size])
    mask_acc = np.zeros([1,opt.seq_size])  # some video has less than 10 segments so we need to mask the rest part

    train_acc = np.zeros([1, opt.seq_size])

    time_start = time.time()
  
    subact_predictions = list()
    subact_ground_truth = list()
    affordance_predictions = list()
    affordance_ground_truth = list()

    #print ("dataloader size is ", len(dataloader))
    # CAD120
    #for i, (edge_features, node_features, adj_mat, node_labels, relation_label, graph_label, sequence_ids, subject_ids) in enumerate(dataloader, 0):
    # sth-sth
    for i, (edge_features, node_features, adj_mat, graph_label,tem_features) in enumerate(dataloader, 0):
        #pdb.set_trace()
        model.zero_grad()  # Sets gradients of all model parameters to zero.
        batch, _, n_node, n_edge = adj_mat.size()
        #print("i=%d"%i)
        adj_mat = adj_mat.contiguous().view(batch*opt.seq_size, n_node, n_edge).float()
        init_input = node_features.contiguous().view(batch*opt.seq_size, n_node, -1).float()
        edge_states = edge_features.contiguous().view(batch*opt.seq_size, n_edge, -1).float()
        
        #print(graph_label.shape)
        
        graph_labels = graph_label[:,:,:12]
        #print(graph_labels.shape)
        true_lengths = graph_label[:,-1:,-1].squeeze().float()  #[batch,1]
        #print(true_lengths)
        if opt.cuda:
            init_input = init_input.cuda()
            #adj_matrixes = adj_mat.cuda()
            edge_states = edge_states.cuda()
            tem_features = tem_features.cuda()
            #relation_label = relation_label.cuda()
            labels = graph_labels.cuda()
            true_lengths = true_lengths.cuda()
            #node_labels = node_labels.cuda()
            #print "all the variable has been put into cuda"
        #print(adj_matrixes)
        #print(init_input)#[48,10,3,512]
        #init_input = init_input[:,:2,:]
        init_input = Variable((init_input))
        #print(init_input.size())
        adj_matrixes = np.zeros((5,25))
        one = np.ones((5,5))
        for j in range(5):
            adj_matrixes[j][j*5:(j+1)*5] = one[j]
        adj_matrixes = torch.Tensor(adj_matrixes).repeat(batch*opt.seq_size, 1, 1)
        adj_matrixes = Variable(adj_matrixes.cuda())
        tem_features = Variable(tem_features)
        #[48,10,25,512]
        '''
        edge_stages = edge_states.contiguous().view(batch*opt.seq_size,5,5,-1)
        edge_stages = edge_stages[:,:2,:2,:]
        edge_stages = edge_stages.contiguous().view(batch*opt.seq_size,4,-1)
        
       
        out = edge_stages[:,:2,:]
        rand = torch.rand((batch*opt.seq_size,2,516)).cuda()
        edge_stages = torch.cat((out,rand),1)
        '''
        edge_states = Variable((edge_states))
        #print(edge_states)
        #print(graph_labels.shape)
        #relation_label = Variable(relation_label)
        labels = Variable(labels)
        true_lengths = Variable(true_lengths)
        #print "All variables have been put in cuda"
        
        if opt.loss==1:
            #print "Using first models."
            pre_out,mse_loss,mmd_loss = model(init_input, edge_states, adj_matrixes, true_lengths, tem_features, is_test=False)
        else:
            pre_out,mse_loss,mmd_loss = model(init_input, edge_states, adj_matrixes, true_lengths, tem_features, is_test=False)
        #pre_out = model(init_input, edge_states)  # for basemodel
        #print(mse_loss,mmd_loss)#########
        pre_label = torch.argmax(pre_out, dim=1)  # [batch*seq_size,1]
        gt_labels = torch.argmax(labels, dim=2)  # [batch,seq_size,1]

        #print(pre_label)
        #print(gt_labels)
        #pre_relation_label = torch.argmax(relation_pre_out, dim=1)
        ##re_gt_labels = torch.argmax(relation_label, dim=1)  # [batch*seq_size,1]

        ##node_labels = torch.argmax(node_labels, dim=1) # [batch*seq_size,node_num,1]
        pre_label = pre_label.view(batch, opt.seq_size, -1)
        gt_labels = gt_labels.view(batch, opt.seq_size, -1)
        #print(gt_labels.squeeze())
        #calculate the accuracy
        for seq_i in range(opt.seq_size):
            correct[0, seq_i] = (pre_label[:, seq_i, 0] == gt_labels[:, seq_i, 0]).sum()
            train_correct[0, seq_i] = correct[0, seq_i] + train_correct[0, seq_i]

        #calculate the mask loss
        gt_labels = gt_labels.long().squeeze()
        pre_out = pre_out.view(batch, opt.seq_size, -1).float()
        true_lengths = true_lengths.long()

        classification_loss = masked_cross_entropy(pre_out,gt_labels,true_lengths)
       
        
        loss =  opt.l1*classification_loss + opt.l2*mse_loss+opt.l3*mmd_loss  # TODO need regularzation??
 
        

        train_loss = train_loss + loss
        mse_losses = mse_losses + mse_loss
        mmd_losses = mmd_losses + mmd_loss
        
        #vis_losses = vis_losses + vis_loss
        #loss = loss.float()
        
        loss.backward()
        optimizer.step()
        #print "train_correct: %d " % train_correct
        if i % opt.batch_size== 0:
            print ("epoch %d, mse_loss:%.4f, mmd_loss:%.4f, classification_loss: %.4f, batch loss: %.4f" % (epoch,mse_loss,mmd_loss, classification_loss, loss))        
    print ("epoch %d, average train loss: %.4f " % (epoch, train_loss / len(dataloader)))
    print ("epoch %d, average mse loss: %.4f " % (epoch, mse_losses / len(dataloader)))
    print ("epoch %d, average mmd loss: %.4f " % (epoch, mmd_losses / len(dataloader)))

    if i % int(len(dataloader) / 10 + 1) == 0 and opt.verbal:
        print ("[%d/%d][%d/%d] Loss: %.4f" % (epoch, opt.niter, i, len(dataloader), loss.data[0]))

    print ('An epoch has trained finished. Time elapsed: {:.2f}s'.format(time.time() - time_start))
    ###train_acc = train_correct / len(dataloader.dataset)

    train_acc = train_correct / len(dataloader.dataset)
    train_loss = train_loss / len(dataloader)
	
    train_mse_loss = mse_losses / len(dataloader)
    train_mmd_loss = mmd_losses / len(dataloader)

    writer.add_scalar('train_loss', train_loss, epoch)
    #writer.add_scalar('vis_loss', vis_losses/ len(dataloader), epoch)


    #log_file.write("epoch %d, average visual loss: %.4f\n\n" % (epoch, vis_losses/ len(dataloader)))

    print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print ("ratio=0.1, train Accuracy: %.2f " % (100. * train_acc[0][0]))
    print ("ratio=0.5, train Accuracy: %.2f " % (100. * train_acc[0][4]))
    print ("ratio=1.0, train Accuracy: %.2f " % (100. * train_acc[0][9]))
    print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    file=open(log_file,'a+')
    file.write("ratio=0.1, train Accuracy: %.2f \n" % (100. * train_acc[0][0]))
    file.write("ratio=0.5, train Accuracy: %.2f \n" % (100. * train_acc[0][4]))
    file.write("ratio=1.0, train Accuracy: %.2f \n" % (100. * train_acc[0][9]))
    file.write("After %d epoch, average train loss: %.4f, average mse loss: %.4f,average mmd loss: %.4f\n\n" % (epoch, train_loss,train_mse_loss,train_mmd_loss))

    return train_acc

