import torch 
import sthv1_model_ablation
import argparse

parser = argparse.ArgumentParser()
# Lin changed 4 to 1 on Sept. 1st
parser.add_argument('--belta', type=int, help='smooth factor', default=10)
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--batch_size', type=int, default=48, help='input batch size')
parser.add_argument('--hidden_size', type=int, default=512, help='hidden size of rnn')
parser.add_argument('--seq_size', type=int, default=10, help='sequence length of rnn')
parser.add_argument('--rnn_num_layer', type=int, default=1, help='layer number of rnn')
parser.add_argument('--classNum', type=int, default=21, help='number of classes')
parser.add_argument('--rclassNum', type=int, default=10, help='number of relation classes')
parser.add_argument('--nodeclassNum', type=int, default=12, help='number of classes')
parser.add_argument('--d_pos', type=int, default=256, help='dimension of position')
parser.add_argument('--max_num_node', type=int, default=5, help='#objects+human')
parser.add_argument('--node_feat_dim', type=int, default=1024, help='node state size')
parser.add_argument('--edge_feat_dim', type=int, default=1028, help='edge state size')
parser.add_argument('--tem_feat_dim', type=int, default=0, help='edge state size')
parser.add_argument('--state_dim', type=int, default=512, help='dim of annotation')
parser.add_argument('--project_dim', type=int, default=256, help='dim of annotation')
parser.add_argument('--semantic_dim', type=int, default=512, help='dim of annotation')
parser.add_argument('--predict_node_num', type=int, default=3, help='number of predicted nodes')
parser.add_argument('--gcn_dim1', type=int, default=512, help='dim of the first gcn layer  in GAE')
parser.add_argument('--gcn_dim2', type=int, default=256, help='dim of the seconde gcn layer in GAE')
parser.add_argument('--dropout', type=float, default=0., help='dropout rate in GAE')
parser.add_argument('--num_bottleneck', type=int, default=256, help='dim of temporal reasoning module')
parser.add_argument('--num_frames', type=int, default=5, help='number of sampled frames in each segment ')
parser.add_argument('--n_steps', type=int, default=3, help='propogation steps number of GGNN')
parser.add_argument('--n_cluster', type=int, default=3, help='number of clusters in K-means')
parser.add_argument('--niter', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--epoch', type=int, default=0, help='index of epoch to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0., help='learning rate')
#parser.add_argument('--resume', default='/home/mcislab/wangruiqi/IJCAI2019/results/something/NoSpatial/ckp',help='path to latest checkpoint')
#parser.add_argument('--logroot', default='/home/mcislab/wangruiqi/IJCAI2019/results/something/NoSpatial/log',help='path to latest checkpoint')
parser.add_argument('--resume', default='/home/mcislab/zhaojw/AAAI/prediction2020/models/sthv1_only_reasoning/1/',help='path to latest checkpoint')
parser.add_argument('--logroot', default='/home/mcislab/zhaojw/AAAI/prediction2020/log/sthv1_only_reasoning/1/',help='path to latest checkpoint')
parser.add_argument('--cuda', action='store_true', default=False, help='enables cuda')
parser.add_argument('--device_id', type=int, default=0, help='device id of gpu')
parser.add_argument('--verbal', action='store_true', help='print training info or not')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--loss', type=int, default=1, help='loss type')
parser.add_argument('--l1', type=int, default=1, help='loss type')
parser.add_argument('--l2', type=int, default=1/2, help='loss type')
parser.add_argument('--l3', type=int, default=1/2, help='loss type')
parser.add_argument('--show', action='store_true', help='show arch')
parser.add_argument('--featdir', type=str, help='feat dir')



opt = parser.parse_args()

print(opt)

model_path = '/home/mcislab/zhaojw/AAAI/prediction2020/models/sthv1_only_reasoning/1/1016/model_best.pth'

checkpoint = torch.load(model_path)





model = sthv1_model_ablation.Model(opt)    

model.load_state_dict_(checkpoint)

exit()
