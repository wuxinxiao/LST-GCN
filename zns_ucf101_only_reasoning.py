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
from utils.zns_ucf101_train import train 
from utils.zns_ucf101_test import test 
from utils.val import val
#from datasets.UCF101dataset_general import dataset
from datasets.UCF101datasets import dataset
import datasets.utils as utils
import datasets.config as config
import zns_ucf101_only_reasoning_model
import datasets
from tensorboardX import SummaryWriter
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#from utils.datasets.ucf11Dataloader import ucf11Dataloader

parser = argparse.ArgumentParser()
# Lin changed 4 to 1 on Sept. 1st
parser.add_argument('--belta', type=int, help='smooth factor', default=10)
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batch_size', type=int, default=48, help='input batch size')
parser.add_argument('--hidden_size', type=int, default=256, help='hidden size of rnn')
parser.add_argument('--seq_size', type=int, default=10, help='sequence length of rnn')
parser.add_argument('--rnn_num_layer', type=int, default=1, help='layer number of rnn')
parser.add_argument('--classNum', type=int, default=101, help='number of classes')
parser.add_argument('--rclassNum', type=int, default=10, help='number of relation classes')
parser.add_argument('--nodeclassNum', type=int, default=12, help='number of classes')
parser.add_argument('--max_num_node', type=int, default=5, help='number of classes')
parser.add_argument('--n_type_node', type=int, default=11, help='#objects+human')
parser.add_argument('--node_feat_dim', type=int, default=2048, help='node state size')  #512#
parser.add_argument('--edge_feat_dim', type=int, default=2052, help='edge state size')  #516#
parser.add_argument('--tem_feat_dim', type=int, default=2048, help='edge state size') #512#
parser.add_argument('--state_dim', type=int, default=512, help='dim of annotation')
parser.add_argument('--project_dim', type=int, default=256, help='dim of annotation')
parser.add_argument('--semantic_dim', type=int, default=512, help='dim of annotation')
parser.add_argument('--n_cluster', type=int, default=3, help='clustering number of object feature')
parser.add_argument('--predict_node_num', type=int, default=3, help='number of predicted nodes')
parser.add_argument('--gcn_dim1', type=int, default=512, help='dim of the first gcn layer  in GAE')
parser.add_argument('--gcn_dim2', type=int, default=512, help='dim of the seconde gcn layer in GAE')
parser.add_argument('--dropout', type=float, default=0., help='dropout rate in GAE')
parser.add_argument('--num_bottleneck', type=int, default=256, help='dim of temporal reasoning module')
parser.add_argument('--num_frames', type=int, default=8, help='number of sampled frames in each segment ')###5
parser.add_argument('--n_steps', type=int, default=3, help='propogation steps number of GGNN')
parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')#50 150 300
parser.add_argument('--epoch', type=int, default=0, help='index of epoch to train for')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')###0.0005
parser.add_argument('--weight_decay', type=float, default=0., help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum of SGD optimizer')
parser.add_argument('--resume', default='/home/mcislab/zhaojw/AAAI/prediction2020/models/ucf101_res50_rgbflow_same/only_reasoning/num_frames8_lr0.0005/',help='path to latest checkpoint')
parser.add_argument('--logroot', default='/home/mcislab/zhaojw/AAAI/prediction2020/log/ucf101_res50_rgbflow_same/only_reasoning/num_frames8_lr0.0005/',help='path to latest log')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--device_id', type=int, default=1, help='device id of gpu')
parser.add_argument('--verbal', action='store_true', help='print training info or not')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--l1', type=int, default=1, help='loss type')
parser.add_argument('--l2', type=int, default=1/8, help='loss type')
parser.add_argument('--l3', type=int, default=1, help='loss type')
parser.add_argument('--split',type=str,default='1',help='the name of root directory')#1#2
parser.add_argument('--ggnn', action='store_true', help='enables cuda')
opt = parser.parse_args()

print(opt)



if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 3000)
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
    #log_dir_name = 'split'+opt.split + '/'
    log_dir_name = 'split'+opt.split + '/'+str(opt.manualSeed)+'/'
    opt.resume = os.path.join(opt.resume,log_dir_name)
    log_path = os.path.join(opt.logroot,log_dir_name)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    #log_file_name = log_path + 'ucf_log_st.txt'
    log_file_name = log_path + 'ucf_log_v5.0_st_'+str(opt.manualSeed)+'.txt'

    with open(log_file_name,'a+') as file:
        file.write('manualSeed is %d \n' % opt.manualSeed)
        file.write('state_dim is %d \n' % opt.state_dim)
        file.write('num_bottleneck is %d \n' % opt.num_bottleneck)
    paths = config.Paths()

    train_datalist = '/home/mcislab/wangruiqi/IJCV2019/data/ucf101Vid_train_lin_split'+opt.split+'.txt'
    test_datalist = '/home/mcislab/wangruiqi/IJCV2019/data/ucf101Vid_val_lin_split'+opt.split+'.txt'
    #train_datalist = '/home/mcislab/wangruiqi/IJCV2019/data/test.txt'
    #test_datalist = '/home/mcislab/wangruiqi/IJCV2019/data/test.txt'
    #train_dataset = dataset(train_datalist, paths.detect_root_ucf_mmdet, paths.img_root_ucf,paths.rgb_res18_ucf,paths.rgb_res18_ucf, opt)
    train_dataset = dataset(train_datalist, paths.resnet50_ucf_rgbflow_same,opt)####zhao changed ###paths.resnet50_ucf_rgbflow_same
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers, drop_last=False)
    #test_dataset = dataset(test_datalist, paths.detect_root_ucf_mmdet, paths.img_root_ucf,paths.rgb_res18_ucf,paths.rgb_res18_ucf, opt)
    test_dataset = dataset(test_datalist, paths.resnet50_ucf_rgbflow_same,opt)###paths.resnet50_ucf_rgbflow_same
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers, drop_last=False)

    model = zns_ucf101_only_reasoning_model.Model(opt)
    print(model)
    #exit()
    #optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    optimizer = optim.SGD(model.parameters(), lr=opt.lr,momentum=opt.momentum)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.9)
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.NLLLoss()

    if opt.cuda:
        
        model.cuda()
        #criterion.cuda(opt.device_id)
        criterion1.cuda()
        criterion2.cuda()

    
    '''
    if opt.epoch != 0:
        if os.path.exists('./models/hmdb_split1/'+checkpoint_model_name):
            model.load_state_dict(torch.load('./models/hmdb_split1/' + checkpoint_model_name))
        else:
            print('model not found')
            exit()
    '''
    #Lin commented on Sept. 2nd
    #model.double()


    writer = SummaryWriter(log_dir=log_path+'runs/')
    # For training
    sum_test_acc = []
    best_acc = 0.
    epoch_errors = list()
    avg_epoch_error = np.inf
    best_epoch_error = np.inf
    '''
    #haha, output Acc for each class
    model.load_state_dict(torch.load('/home/mcislab/linhanxi/ucf101_NewFeat_RGBtuned/ckpnothresh/ours/model_best.pth')['state_dict'])
    test_acc, output = test(0,test_dataloader, model, criterion1, criterion2, opt, writer, log_file_name, is_test=True)
    exit()
    
    #load last experiment best model
    print("load last experiment best model")
    model.load_state_dict(torch.load('/home/mcislab/zhaojw/AAAI/prediction2020/models/ucf101_res50/split1/1050/model_best.pth')['state_dict'])
    test_acc, output = test(0,test_dataloader, model, criterion1, criterion2, opt, writer, log_file_name, is_test=True)
    '''
    
    print ("Test once for a baseline.")
    loaded_checkpoint =utils.load_best_checkpoint(opt, model, optimizer)
    if loaded_checkpoint:
        #opt, model, optimizer = loaded_checkpoint
        opt, model, __ = loaded_checkpoint
        test_acc, output = test(1,test_dataloader, model, criterion1, criterion2, opt, writer, log_file_name, is_test=True)
        tmp_test_acc = np.mean(test_acc)
        if tmp_test_acc > best_acc:
         
            best_acc = tmp_test_acc
        

    print ("Start to train.....")
    #model.load_state_dict(torch.load('/home/mcislab/linhanxi/ucf101_flowOnly/ckpnothresh/ours/checkpoint.pth')['state_dict'])
    for epoch_i in range(opt.epoch, opt.niter):
        scheduler.step()
        
        train(epoch_i, train_dataloader, model, criterion1, criterion2,  optimizer, opt, writer, log_file_name)
        #val_acc, val_out, val_error =test(valid_loader, model, criterion1,criterion2, opt, log_file_name, is_test=False)
        # Lin changed according to 'sth_pre_abl1' on Sept. 3rd
        test_acc, output = test(epoch_i,test_dataloader, model, criterion1, criterion2, opt, writer, log_file_name, is_test=True)
        #test_acc,_ = test(test_dataloader, model, criterion1, criterion2, opt, log_file_name, is_test=True)
        
        tmp_test_acc = np.mean(test_acc)
        sum_test_acc.append(test_acc)
     
        if tmp_test_acc > best_acc:
            is_best = True
            best_acc = tmp_test_acc

        else:
            is_best = False

        utils.save_checkpoint({'epoch': epoch_i , 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                              is_best=is_best, directory=opt.resume)
        print ("A training epoch finished!")
       
    # For testing
   
    print ("Training finished.Start to test.")
    loaded_checkpoint = utils.load_best_checkpoint(opt, model, optimizer)
    if loaded_checkpoint:
        opt, model, __ = loaded_checkpoint
    # Lin changed according to 'sth_pre_abl1' on Sept. 3rd
    test_acc,output = test(epoch_i,test_dataloader, model, criterion1, criterion2, opt, writer, log_file_name, is_test=True)
    #test_acc,output = test(test_dataloader, model, criterion1,criterion2,  opt, log_file_name, is_test=True)
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
    #sum_test_acc = np.array(sum_test_acc)
    #sum_test_acc=sum_test_acc.reshape(opt.niter-opt.epoch, opt.seq_size)
    #scio.savemat(log_path+'result_st.mat',{'test_acc':sum_test_acc})
    #scio.savemat(log_path+'result_st_output.mat',{'test_out':output})

if __name__ == "__main__":
    main(opt)
