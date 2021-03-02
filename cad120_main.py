"""
Created on Feb 21, 2019

@author: Wang Ruiqi

Description of the file.


version3.0

"""
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
from utils.cad_train import train
from utils.cad_test import test
import datasets.CAD120.cad120_config as cad120_config
from datasets.CAD120.cad120 import CAD120
import datasets.utils as utils

import models
from tensorboardX import SummaryWriter


#from utils.datasets.ucf11Dataloader import ucf11Dataloader

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
parser.add_argument('--hidden_size', type=int, default=512, help='hidden size of rnn')
parser.add_argument('--seq_size', type=int, default=10, help='sequence length of rnn')
parser.add_argument('--rnn_num_layer', type=int, default=1, help='layer number of rnn')
parser.add_argument('--classNum', type=int, default=10, help='number of classes')
parser.add_argument('--rclassNum', type=int, default=10, help='number of relation classes')
parser.add_argument('--nodeclassNum', type=int, default=12, help='number of classes')
parser.add_argument('--n_nodes', type=int, default=6, help='number of nodes in graph')
parser.add_argument('--d_pos', type=int, default=256, help='dimension of position')
parser.add_argument('--belta', type=int, default=10, help='smoothing factor in vlad')
parser.add_argument('--node_feat_dim', type=int, default=810, help='node state size')
parser.add_argument('--edge_feat_dim', type=int, default=800, help='edge state size')
parser.add_argument('--state_dim', type=int, default=512, help='dim of annotation')
parser.add_argument('--tem_feat_dim', type=int, default=0, help='edge state size')
parser.add_argument('--project_dim', type=int, default=256, help='dim of annotation')
parser.add_argument('--semantic_dim', type=int, default=512, help='dim of annotation')
parser.add_argument('--predict_node_num', type=int, default=3, help='number of predicted nodes')
parser.add_argument('--gcn_dim1', type=int, default=512, help='dim of the first gcn layer  in GAE')
parser.add_argument('--gcn_dim2', type=int, default=256, help='dim of the seconde gcn layer in GAE')
parser.add_argument('--dropout', type=float, default=0., help='dropout rate in GAE')
parser.add_argument('--num_bottleneck', type=int, default=256, help='dim of temporal reasoning module')
parser.add_argument('--num_frames', type=int, default=4, help='number of sampled frames in each segment ')
parser.add_argument('--n_steps', type=int, default=3, help='propogation steps number of IGGNN')
parser.add_argument('--n_cluster', type=int, default=3, help='number of clusters in K-means')
parser.add_argument('--niter', type=int, default=300, help='number of epochs to train for')
parser.add_argument('--epoch', type=int, default=0, help='index of epoch to train for')
parser.add_argument('--lr', type=float, default=0.00005, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0., help='learning rate')
parser.add_argument('--resume', default='/home/mcislab/wangruiqi/prediction2020/model/cad/',help='path to latest checkpoint')
parser.add_argument('--logroot', default='/home/mcislab/wangruiqi/prediction2020/log/cad/',help='path to latest checkpoint')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--device_id', type=int, default=0, help='device id of gpu')
parser.add_argument('--verbal', action='store_true', help='print training info or not')
parser.add_argument('--manualSeed', type=int, default=5002,help='manual seed')
parser.add_argument('--l1', type=int, default=1, help='loss type')
parser.add_argument('--l2', type=int, default=1/2, help='loss type')
parser.add_argument('--l3', type=int, default=1/2, help='loss type')
parser.add_argument('--subject', type=int, default=1, help='subject index')
parser.add_argument('--featdir', type=str, help='feat dir')
opt = parser.parse_args()

print(opt)
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 5000)
#print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

def main(opt):
    if not os.path.exists(opt.resume):
        os.makedirs(opt.resume)
    if not os.path.exists(opt.logroot):
        os.makedirs(opt.logroot)

  
    paths = cad120_config.Paths()
    #print (paths.tmp_root)
    subject_ids = pickle.load(open(os.path.join(paths.tmp_root, 'cad120_data_list.p'), 'rb'))

    data_path = os.path.join(paths.tmp_root, 'cad120_data_pred.p')
    #data_path =  '/media/mcislab/wrq/CAD120/pred-feature/cad120_data_pred.p'
    test_acc_final = np.zeros([4, opt.seq_size])
    sub_index = 0
    
    resume_root = os.path.join(opt.resume,str(opt.manualSeed))
    
    for sub, seqs in subject_ids.items():  # cross-validation for each subject
        #sub='Subject'+str(opt.subject)
        log_file_name = opt.logroot+'cad120_log_sub'+sub+'.txt'
        with open(log_file_name,'a+') as file:
            file.write('manualSeed is %d \n' % opt.manualSeed)
        opt.resume = resume_root+ sub + '/'

        training_subject = pickle.load(open(os.path.join(paths.tmp_root, 'cad120_data_list.p'), 'rb'))  # if not reload it will delete both in subject_ids and training_sub
        testing_subject = dict()
        testing_subject[sub] = seqs
        
        del training_subject[sub]

        
        #print training_subject
        #print testing_subject

        training_set = CAD120(data_path, training_subject)

        testing_set = CAD120(data_path, testing_subject)

        #testing_set = CAD120(data_path, sequence_ids[-test_num:])


        train_loader = torch.utils.data.DataLoader(training_set, collate_fn=utils.collate_fn_cad,batch_size=opt.batch_size,
                                                   num_workers=opt.workers, shuffle=True, pin_memory=True)
        #valid_loader = torch.utils.data.DataLoader(valid_set, collate_fn=utils.collate_fn_cad,
        #                                           batch_size=opt.batch_size,
        #                                           num_workers=opt.workers, shuffle=False, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(testing_set,collate_fn=utils.collate_fn_cad,batch_size=opt.batch_size,
                                                  num_workers=opt.workers, shuffle=False, pin_memory=True)

        model = models.Model(opt) 
        optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
        criterion1 = nn.CrossEntropyLoss()
        criterion2 = nn.NLLLoss()
        if opt.cuda:
            model.cuda()
            #criterion.cuda(opt.device_id)
            criterion1.cuda()
            criterion2.cuda()


        loaded_checkpoint = utils.load_best_checkpoint(opt, model, optimizer)
        if loaded_checkpoint:
            opt, model, optimizer = loaded_checkpoint
        '''
        if opt.epoch != 0:
            if os.path.exists('./models/hmdb_split1/'+checkpoint_model_name):
                model.load_state_dict(torch.load('./models/hmdb_split1/' + checkpoint_model_name))
            else:
                print('model not found')
                exit()
        '''
        #model.double()


        writer = SummaryWriter(log_dir=opt.logroot+'runs/'+sub+'/')
        # For training
        sum_test_acc = []
        best_acc = 0.
        epoch_errors = list()
        avg_epoch_error = np.inf
        best_epoch_error = np.inf
        
        for epoch_i in range(opt.epoch, opt.niter):
            scheduler.step()
            train(epoch_i, train_loader, model, criterion1,criterion2,  optimizer, opt, writer, log_file_name)
            #val_acc, val_out, val_error =test(valid_loader, model, criterion1,criterion2, opt, log_file_name, is_test=False)
            test_acc,output = test(epoch_i,test_loader, model, criterion1, criterion2, opt,writer, log_file_name)
            
            tmp_test_acc = np.mean(test_acc)
            sum_test_acc.append(test_acc)
            
            if tmp_test_acc > best_acc:
                is_best = True
                best_acc = tmp_test_acc


            else:
                is_best = False

            utils.save_checkpoint({'epoch': epoch_i + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                                  is_best=is_best, directory=opt.resume)
        
        # For testing
        loaded_checkpoint = utils.load_best_checkpoint(opt, model, optimizer)
        if loaded_checkpoint:
            opt, model, optimizer = loaded_checkpoint
        test_acc,output = test(epoch_i,test_loader, model, criterion1, criterion2, opt,writer, log_file_name)
        print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print ("ratio=0.1, test Accuracy:   %.2f " % (100. * test_acc[0][0]))
        print ("ratio=0.2, test Accuracy:   %.2f " % (100. * test_acc[0][1]))
        print ("ratio=0.3, test Accuracy:   %.2f " % (100. * test_acc[0][2]))
        print ("ratio=0.4, test Accuracy:   %.2f " % (100. * test_acc[0][3]))
        print ("ratio=0.5, test Accuracy:   %.2f " % (100. * test_acc[0][4]))
        print ("ratio=0.6, test Accuracy:   %.2f " % (100. * test_acc[0][5]))
        print ("ratio=0.7, test Accuracy:   %.2f " % (100. * test_acc[0][6]))
        print ("ratio=0.8, test Accuracy:   %.2f " % (100. * test_acc[0][7]))
        print ("ratio=0.9, test Accuracy:   %.2f " % (100. * test_acc[0][8]))
        print ("ratio=1.0, test Accuracy:   %.2f " % (100. * test_acc[0][9]))
        print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        sum_test_acc = np.array(sum_test_acc)
        sum_test_acc=sum_test_acc.reshape(opt.niter, opt.seq_size)
        scio.savemat(opt.logroot+sub+'_result.mat',{'test_acc':sum_test_acc})
        scio.savemat(opt.logroot+sub+'_output.mat',{'test_out':output})

        test_acc_final[sub_index, :] = test_acc
        
        with open(log_file_name, 'a+') as file:
            file.write("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
            file.write("ratio=0.1, test Accuracy: %.2f \n" % (100. * test_acc[0][0]))
            file.write("ratio=0.2, test Accuracy: %.2f \n" % (100. * test_acc[0][1]))
            file.write("ratio=0.3, test Accuracy: %.2f \n" % (100. * test_acc[0][2]))
            file.write("ratio=0.4, test Accuracy: %.2f \n" % (100. * test_acc[0][3]))
            file.write("ratio=0.5, test Accuracy: %.2f \n" % (100. * test_acc[0][4]))
            file.write("ratio=0.6, test Accuracy: %.2f \n" % (100. * test_acc[0][5]))
            file.write("ratio=0.7, test Accuracy: %.2f \n" % (100. * test_acc[0][6]))
            file.write("ratio=0.8, test Accuracy: %.2f \n" % (100. * test_acc[0][7]))
            file.write("ratio=0.9, test Accuracy: %.2f \n" % (100. * test_acc[0][8]))
            file.write("ratio=1.0, test Accuracy: %.2f \n" % (100. * test_acc[0][9]))
        sub_index = sub_index + 1
        writer.close()


    test_final = np.mean(test_acc_final,0)
    #print type(test_final)
    #print test_final
    print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print ("ratio=0.1, test Accuracy:   %.2f " % (100. * test_final[0]))
    print ("ratio=0.2, test Accuracy:   %.2f " % (100. * test_final[1]))
    print ("ratio=0.3, test Accuracy:   %.2f " % (100. * test_final[2]))
    print ("ratio=0.4, test Accuracy:   %.2f " % (100. * test_final[3]))
    print ("ratio=0.5, test Accuracy:   %.2f " % (100. * test_final[4]))
    print ("ratio=0.6, test Accuracy:   %.2f " % (100. * test_final[5]))
    print ("ratio=0.7, test Accuracy:   %.2f " % (100. * test_final[6]))
    print ("ratio=0.8, test Accuracy:   %.2f " % (100. * test_final[7]))
    print ("ratio=0.9, test Accuracy:   %.2f " % (100. * test_final[8]))
    print ("ratio=1.0, test Accuracy:   %.2f " % (100. * test_final[9]))
    print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    with open(log_file_name, 'a+') as file:
        file.write("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
        file.write("Cross-subject performance is:\n")
        file.write("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
        file.write("ratio=0.1, test Accuracy: %.2f \n" % (100. * test_final[0]))
        file.write("ratio=0.2, test Accuracy: %.2f \n" % (100. * test_final[1]))
        file.write("ratio=0.3, test Accuracy: %.2f \n" % (100. * test_final[2]))
        file.write("ratio=0.4, test Accuracy: %.2f \n" % (100. * test_final[3]))
        file.write("ratio=0.5, test Accuracy: %.2f \n" % (100. * test_final[4]))
        file.write("ratio=0.6, test Accuracy: %.2f \n" % (100. * test_final[5]))
        file.write("ratio=0.7, test Accuracy: %.2f \n" % (100. * test_final[6]))
        file.write("ratio=0.8, test Accuracy: %.2f \n" % (100. * test_final[7]))
        file.write("ratio=0.9, test Accuracy: %.2f \n" % (100. * test_final[8]))
        file.write("ratio=1.0, test Accuracy: %.2f \n" % (100. * test_final[9]))

if __name__ == "__main__":
    main(opt)
