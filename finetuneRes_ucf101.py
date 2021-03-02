import torch
import torch.nn as nn
from torchvision import models, transforms, utils
from skimage import io, transform
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
#from MotionMapFrame import Rescale, ToTensor, TensorNormalize, MotionMapFrame
from UCF101Dataset import Rescale, ToTensor, TensorNormalize, UCF101Dataset
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def save_net(model, epoch, i, acc):
    if not os.path.exists('/home/mcislab/linhanxi/IJCV19/ucf101_res18_model/'):
        os.makedirs('/home/mcislab/linhanxi/IJCV19/ucf101_res18_model/')
    torch.save(model.state_dict(), '/home/mcislab/linhanxi/IJCV19/ucf101_res18_model/epoch%d_i%d_acc%f.pkl' %(epoch, i, acc))

def adjust_learning_rate(optimizer):

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * 0.1

def load_net(net, model_name):
    if os.path.exists('/home/mcislab/linhanxi/IJCV19/ucf101_res18_model/' + model_name):
        net.load_state_dict(torch.load('/home/mcislab/linhanxi/IJCV19/ucf101_res18_model/' + model_name))
    else:
        print('no model')

def fintune(model, train_dataloader, test_dataloader, op_lr = 0.001, op_momentum = 0.9):

    criterion = nn.CrossEntropyLoss()
    #ignored_params = list(map(id, model.fc.parameters()))
    #base_params = filter(lambda p: id(p) not in ignored_params,
    #                     model.parameters())
    optimizer = torch.optim.SGD(model.parameters(), lr=op_lr , momentum=op_momentum)
    #optimizer = torch.optim.SGD(model.fc.parameters(),lr = op_lr, momentum = op_momentum)
    #optimizer = torch.optim.Adam([{'params' : base_params}, {'params' : model.fc.parameters(), 'lr' : op_lr}], lr = op_lr*0.1 )
    print('Begin epoch!')
    for epoch in range(500):
        running_loss = 0.0
        # class_correct = list(0. for i in range(101))
        # class_total = list(0. for i in range(101))
        if(epoch % 10 == 9):
            adjust_learning_rate(optimizer)

        for i, data in enumerate(train_dataloader):
            input, label,path = data

            if torch.cuda.is_available():
                labels = label.cuda()
                # input, label = Variable(input.cuda()), Variable(label.cuda())
                input, label = Variable(input.cuda()), Variable(labels)
                
            else:
                labels = label
                input, label = Variable(input), Variable(label)
                
            optimizer.zero_grad()
            output = model(input)
            
            #out = model(Variable(input))
            
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            # running_loss += loss.data[0]
            running_loss += loss.item()
            
            if( i % 10 == 9):
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0
                # print ('\nTrain Acc : %.4f %%\n' % train_acc)
            
            if( i % 1000 == 500):
                # print('[%d, %5d] loss: %.3f' %
                      # (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
                train_acc = test(model, train_dataloader)
                print ('\nTrain Acc : %.4f %%\n' % train_acc)
            # if( i % 5000 == 4999):
            if( i % 1000 == 999):
                print ('haha')
                #train_acc = test(model, train_dataloader)
                acc = test(model, test_dataloader)
                print ('\nTest Acc : %.4f %%\n' % acc)
                save_net(model,i + 1, i, acc)
        
        train_acc = test(model, train_dataloader)
        print ('\nTrain Acc : %.4f %%\n' % train_acc)
        acc = test(model, test_dataloader)
        save_net(model, epoch + 1, 0, acc)
        print ('\nTest Acc : %.4f %%\n' % acc)

def test(model, test_loader):

    # class_correct = list(0. for i in range(101))
    # class_total = list(0. for i in range(101))
    # class_total = range(101)
    # class_correct = range(101)
    acc = 0.0
    # for data in test_loader:
    for i, data in enumerate(test_loader):
        images, labels ,path= data
        # print labels.size()[0]
        # model.cuda()
        # model.eval()
        if torch.cuda.is_available():
            out = model(Variable(images.cuda()))
            labels = labels.cuda()
        else:
            out = model(Variable(images))
        # model.train(False)
        # print out.size()
        _, predicted = torch.max(out.data, 1)
        # print labels[0:10]
        # print torch.sum(predicted == labels)
        # print max(predicted)
        
        # c = (predicted == labels).squeeze()
        nins = labels.size()[0]
        # print nins
        # nins = (nins*1.).cuda()
        c = torch.sum(predicted == labels).type(torch.cuda.FloatTensor)/(nins*1.)
        # print c
        if( i % 100 == 99):
          print ('.'),
        # print (c)
        acc += c.item()
        if ( i > 1000):
          break
    acc = acc/float((i+1))
    # print ('\nTrain Acc : %.4f %%\n' % acc)

        # ll = len(labels)
        

        # for i in range(ll):
            # label = labels[i]
            # class_correct[label] += c[i]
            # class_total[label] += 1
        

    # answer = 0.0
    # for i in range(101):
    #    print('Accuracy of %d : %2d %%' % (
    #        i, 100 * class_correct[i] / class_total[i] if not (class_total[i] == 0) else 0))
        # answer += 100. * float(class_correct[i]) / float(class_total[i])
        #print "i=%d"%i
        #print class_total[i]
        #print class_correct[i]
    #print ('\nTest Acc : %.4f %%\n\n' %(answer / 101))
    # return answer / 101.
    return acc*100.
if __name__ == '__main__':


    #model = models.resnet152(pretrained = True)
    model = models.resnet18(pretrained = True)
    model.fc = torch.nn.Linear(512, 101)
    load_net(model, 'epoch6000_i5999_acc89.853106.pkl')
    model.cuda()
    list_dir = ['test_list_1', 'train_list_1']
    # composed = transforms.Compose(
        # [Rescale(224),ToTensor(), TensorNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
 
    composed = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                              std = [ 0.229, 0.224, 0.225 ])])
                              
    for cnt in range(1):
        #cnt = cnt + 2
        #train_dataset = UTInteractionDataset(list_file = './data/list/train_list_%d' %(cnt + 1), root_dir='', class_file='./data/list/classInd.txt', transform=composed)
        train_dataset = UCF101Dataset(list_file = '/home/mcislab/linhanxi/IJCV19/ucf101_train_lin2.txt', root_dir='', class_file='/home/mcislab/linhanxi/IJCV19/ucf101_classInd.txt', transform=composed)
        print('len of train_dataset', len(train_dataset))
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=1)
        #test_dataset = UTInteractionDataset(list_file = './data/list/test_list_%d' %(cnt + 1), root_dir='', class_file='./data/list/classInd.txt', transform=composed)
        test_dataset = UCF101Dataset(list_file = '/home/mcislab/linhanxi/IJCV19/ucf101_val_lin2.txt', root_dir='', class_file='/home/mcislab/linhanxi/IJCV19/ucf101_classInd.txt', transform=composed)
        print('len of test_dataset', len(test_dataset))
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=1)

        print('Begin training!!')

        fintune(model, train_dataloader,test_dataloader)
