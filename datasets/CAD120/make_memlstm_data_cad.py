"""
Created on Mar 13, 2017

@author: Siyuan Qi

Description of the file.

"""

import os
import time
import pickle

import numpy as np

import cad120_config
import metadata

global MAX_EDGE_NUM 
global MAX_NODE_NUM
MAX_EDGE_NUM = 0
MAX_NODE_NUM = 0
def parse_colon_seperated_features(colon_seperated):
    f_list = [int(x.split(':')[1]) for x in colon_seperated]
    return f_list


def read_features(segments_feature_path, filename):
    data = dict()
    filename_base = os.path.basename(filename)
    sequence_id = filename_base.split('_')[0]
    segment_index = int(os.path.splitext(filename_base)[0].split('_')[1])

    # Spatial features
    with open(filename) as f:
        first_line = f.readline().strip()
        object_num = int(first_line.split(' ')[0])
        object_object_num = int(first_line.split(' ')[1])
        skeleton_object_num = int(first_line.split(' ')[2])
        subactivities_label_num = int(first_line.split(' ')[4])
        activities_label_num = 10
        #edge_features = np.zeros((1+object_num, 1+object_num, 800))
        edge_num = (1+object_num)* (1+object_num)
        node_num = 1+object_num
        edge_features = np.zeros((edge_num, 800))
        node_features = np.zeros((1+object_num, 810))     
        global MAX_EDGE_NUM
        global MAX_NODE_NUM 
        if edge_num > MAX_EDGE_NUM:
            
        	MAX_EDGE_NUM = edge_num
        if node_num > MAX_NODE_NUM:
           
            MAX_NODE_NUM = node_num
        # Object feature
        for _ in range(object_num):
            line = f.readline()
            colon_seperated = [x.strip() for x in line.strip().split(' ')]
            o_id = int(colon_seperated[1])
            
            node_features[o_id, 630:] = np.array(parse_colon_seperated_features(colon_seperated[2:]))

        # Skeleton feature
        line = f.readline()

        colon_seperated = [x.strip() for x in line.strip().split(' ')]
   
        node_features[0, :630] = parse_colon_seperated_features(colon_seperated[2:])

        # Object-object feature
        for _ in range(object_object_num):
            line = f.readline()
            colon_seperated = [x.strip() for x in line.strip().split(' ')]
            o1_id, o2_id = int(colon_seperated[2]), int(colon_seperated[3])

            edge_features[o1_id*(1+object_num)+o2_id, 400:600] = parse_colon_seperated_features(colon_seperated[4:])

        # Skeleton-object feature
        for _ in range(skeleton_object_num):
            line = f.readline()
            colon_seperated = [x.strip() for x in line.strip().split(' ')]
            s_o_id = int(colon_seperated[2])
            #edge_features[0,s_o_id, :400] = parse_colon_seperated_features(colon_seperated[3:])
            edge_features[s_o_id, :400] = parse_colon_seperated_features(colon_seperated[3:])
            #edge_features[s_o_id, 0, :400] = edge_features[0, s_o_id, :400]
            edge_features[s_o_id*(1+object_num), :400] = edge_features[s_o_id, :400]
         
    # Temporal features
    if segment_index == 1:
        for node_i in range((1+object_num)):
            #edge_features[node_i, node_i, 600:] = 0
            edge_features[node_i*(1+object_num)+node_i, 600:] = 0
    else:
        with open(os.path.join(segments_feature_path, '{}_{}_{}.txt'.format(sequence_id, segment_index-1, segment_index)), 'r') as f:
            first_line = f.readline().strip()
            object_object_num = int(first_line.split(' ')[0])
            skeleton_skeleton_num = int(first_line.split(' ')[1])
            assert skeleton_skeleton_num == 1

            # Object-object temporal feature
            for _ in range(object_object_num):
                line = f.readline()
                colon_seperated = [x.strip() for x in line.strip().split(' ')]
                o_id = int(colon_seperated[2])
                #edge_features[o_id, o_id, 760:] = np.array(parse_colon_seperated_features(colon_seperated[3:]))
                edge_features[o_id*(1+object_num)+o_id, 760:] = np.array(parse_colon_seperated_features(colon_seperated[3:]))

            # Skeleton-object temporal feature
            line = f.readline()
            colon_seperated = [x.strip() for x in line.strip().split(' ')]
            node_features[0, 600:760] = parse_colon_seperated_features(colon_seperated[3:])

    # Return data as a dictionary
    
    data['edge_features'] = edge_features
    data['node_features'] = node_features
    return data


def collect_data(paths):
    if not os.path.exists(paths.tmp_root):
        os.makedirs(paths.tmp_root)

    segments_files_path = os.path.join(paths.data_root, 'features_cad120_ground_truth_segmentation', 'segments_svm_format')
    segments_feature_path = os.path.join(paths.data_root, 'features_cad120_ground_truth_segmentation', 'features_binary_svm_format')

    data = dict()
    sequence_ids = list()
    # date_selection = ['1204142227', '0510175411']


    for sequence_path_file in os.listdir(segments_files_path):
        sequence_id = os.path.splitext(sequence_path_file)[0]
        # if sequence_id not in date_selection:
        #     continue
        data[sequence_id] = list()
        sequence_ids.append(sequence_id)

        with open(os.path.join(segments_files_path, sequence_path_file)) as f:
            first_line = f.readline()
            segment_feature_num = int(first_line.split(' ')[0])
            lines = f.readlines()

            for segment_i in range(segment_feature_num-1):
                current_segment_feature_filename = lines[segment_i].strip()
                next_segment_feature_filename = lines[segment_i+1].strip()
                current_segment_data = read_features(segments_feature_path, os.path.join(segments_feature_path, os.path.basename(current_segment_feature_filename)))
                next_segment_data = read_features(segments_feature_path, os.path.join(segments_feature_path, os.path.basename(next_segment_feature_filename)))
                data[sequence_id].append(current_segment_data)


    #  graph label
    subject_ids = dict()
    count=0
    #train_list = '/media/mcislab/wrq/CAD120/mem_feature/new/train_list4.txt'
    #test_list = '/media/mcislab/wrq/CAD120/mem_feature/new/test_list4.txt'
    for annotation_path_file in os.listdir(paths.anno_root):
        subjects_path = os.path.join(paths.anno_root, annotation_path_file)
        subject_name = annotation_path_file.split('_')[0]
        print subject_name
        subject_index = metadata.subject_index[subject_name]
        #print subject_index
        subject_ids[subject_name] = list()
        num = 0
        label = np.zeros((11,1))
        for category in os.listdir(subjects_path):
            activitylabel = category
            video_label = metadata.activity_index[activitylabel]
            label[:-1,0] = video_label
            #print one_hot_label
            file = os.path.join(subjects_path,category,'activityLabel.txt')
            with open(file,'r') as f:
                
                lines=f.readlines()
                for line in lines:
                    edge_feat = np.zeros((10,MAX_EDGE_NUM, 800))
                    node_feat = np.zeros((10,MAX_NODE_NUM, 810))
                    count = count+1
                    sequence_id = line.split(',')[0]
                    '''
                    if subject_name == 'Subject5':
                        with open(test_list,'a+') as file:
                            file.write(sequence_id+'\n')
                    else:                       
                        with open(train_list,'a+') as file:
                            file.write(sequence_id+'\n')
                    '''
                    #print sequence_id
                    
                    subject_ids[subject_name].append(sequence_id)

                    true_length = len(data[sequence_id][:])
                    label[-1,0] = true_length
                    if true_length >=10:
                        slt_frms = np.ceil(np.arange(0.1,1.1,0.1)*true_length)
                    
                        for i in range(10):
        
                            edge_num = data[sequence_id][int(slt_frms[i])-1]['edge_features'].shape[0]
                            edge_feat[i,:edge_num] = data[sequence_id][int(slt_frms[i])-1]['edge_features']
                            node_num = data[sequence_id][int(slt_frms[i])-1]['node_features'].shape[0]
                            node_feat[i,:node_num] = data[sequence_id][int(slt_frms[i])-1]['node_features']
                    else: #copy the feature at the end
                        edge_num,edge_feature_len = data[sequence_id][0]['edge_features'].shape
                        node_num,node_feature_len = data[sequence_id][0]['node_features'].shape
                        for i in range(true_length):
                            edge_feat[i,:edge_num] = data[sequence_id][i]['edge_features']
                            node_feat[i,:node_num] = data[sequence_id][i]['node_features']                            
                        for i in range(true_length,10):
                            edge_feat[i,:edge_num] = data[sequence_id][-1]['edge_features']
                            node_feat[i,:node_num] = data[sequence_id][-1]['node_features']
                                   

                    '''    
                    else:  # copy thr feature in the middle
                        index_primal = np.arange(1.0, true_length + 1, 1.0)
                        new_index = np.round(index_primal / true_length * 10)
                        slt_frm_index = new_index - 1
                        slt_frm_index = slt_frm_index.astype(np.int32)
                        edge_num,edge_feature_len = data[sequence_id][0]['edge_features'].shape
                        node_num,node_feature_len = data[sequence_id][0]['node_features'].shape
                        edge_features_mask = np.ones(( slt_frm_index[0], edge_num, edge_feature_len ))*data[sequence_id][0]['edge_features']
                        node_features_mask = np.ones(( slt_frm_index[0], node_num, node_feature_len ))*data[sequence_id][0]['node_features']
                        edge_feat[ :slt_frm_index[0], :edge_num, :] = edge_features_mask
                        node_feat[ :slt_frm_index[0], :node_num, :] = node_features_mask
                        for ind in range(len(slt_frm_index) - 1):
                            edge_features_mask = np.ones((slt_frm_index[ind+1]-slt_frm_index[ind], edge_num, edge_feature_len )) * data[sequence_id][ind]['edge_features']
                            node_features_mask = np.ones(( slt_frm_index[ind+1]-slt_frm_index[ind], node_num, node_feature_len )) * data[sequence_id][ind]['node_features']
                            edge_num = data[sequence_id][ind]['edge_features'].shape[0]
                            node_num = data[sequence_id][ind]['node_features'].shape[0]
                            #print edge_num
                            #print slt_frm_index[ind]
                            #print slt_frm_index[ind+1]
                            edge_feat[slt_frm_index[ind]:slt_frm_index[ind+1], :edge_num, :] = edge_features_mask
                            node_feat[ slt_frm_index[ind]:slt_frm_index[ind+1], :node_num, :] = node_features_mask
                    '''
                    pickle.dump(label, open(os.path.join('/media/mcislab/wrq/CAD120/mem_feature/new/label/',sequence_id+'.p'), 'wb'))
                    pickle.dump(edge_feat, open(os.path.join('/media/mcislab/wrq/CAD120/mem_feature/new/edge_feat/',sequence_id+'.p'), 'wb'))
                    pickle.dump(node_feat, open(os.path.join('/media/mcislab/wrq/CAD120/mem_feature/new/node_feat/',sequence_id+'.p'), 'wb'))
                    
                    num = num+1
                    count = count+1
        print subject_name," has ",num," samples"

    print "%d samples",count

    #pickle.dump(data, open(os.path.join(paths.tmp_root, 'cad120_data_pred.p'), 'wb'))
    #pickle.dump(subject_ids, open(os.path.join(paths.tmp_root, 'cad120_data_list.p'), 'wb'))
    #
def main():
    paths = cad120_config.Paths()
    start_time = time.time()
    collect_data(paths)
    print('Time elapsed: {:.2f}s'.format(time.time() - start_time))


if __name__ == '__main__':
    main()