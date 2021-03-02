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
        edge_features = np.zeros((edge_num, 800))
        node_features = np.zeros((1+object_num, 810))
        # adj_mat = np.zeros((1+object_num, 1+object_num)) if segment_index == 1 else np.eye(1+object_num)

        adj_mat = np.zeros((1+object_num, edge_num))  # adj_mat for GraphNet
        ##adj_mat = np.zeros((1+object_num, (1+object_num)*edge_num))  # adj_mat for GGNN
        node_labels = np.ones((1+object_num)) * -1
        one_hot_relation_label = np.zeros((1,subactivities_label_num))
        one_hot_graph_label = np.zeros((1,activities_label_num))


        stationary_index = metadata.affordance_index['stationary']
        null_index = metadata.subactivity_index['null']

        # Object feature
        for _ in range(object_num):
            line = f.readline()
            colon_seperated = [x.strip() for x in line.strip().split(' ')]
            o_id = int(colon_seperated[1])
            node_labels[o_id] = int(colon_seperated[0]) - 1
            node_features[o_id, 630:] = np.array(parse_colon_seperated_features(colon_seperated[2:]))

        # Skeleton feature
        line = f.readline()

        one_hot_relation_label[0, int(line.strip().split(' ')[0])-1] = 1
        colon_seperated = [x.strip() for x in line.strip().split(' ')]
        node_labels[0] = int(colon_seperated[0]) - 1
        node_features[0, :630] = parse_colon_seperated_features(colon_seperated[2:])

        # Object-object feature
        for _ in range(object_object_num):
            line = f.readline()
            colon_seperated = [x.strip() for x in line.strip().split(' ')]
            o1_id, o2_id = int(colon_seperated[2]), int(colon_seperated[3])

            if int(node_labels[o1_id]) != stationary_index and int(node_labels[o2_id]) != stationary_index:
                #adj_mat[o1_id, o2_id] = 1
            #print filename_base
            #print o1_id
            #print o2_id
            #print object_num

                adj_mat[o1_id][o1_id*(1+object_num)+o2_id] = 1  # for GraphNet
                ## for GGNN
                ##src_idx = o1_id
                ##e_type = o1_id*(1+object_num)+o2_id
                ##tgt_idx = o2_id
                ##adj_mat[tgt_idx - 1][(e_type - 1) * (1+object_num) + src_idx - 1] = 1
                #print filename_base
                #print o1_id
                #print o2_id




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

            if int(node_labels[0]) != null_index and int(node_labels[s_o_id]) != stationary_index:
                #adj_mat[0, s_o_id] = 1
                #adj_mat[s_o_id, 0] = 1
                adj_mat[0, s_o_id] = 1  # for GraphNet
                ##adj_mat[0, s_o_id*(1+object_num)] = 1

                ## for GGNN
                ##src_idx = 0
                ##e_type = s_o_id
                ##tgt_idx = s_o_id
                ##adj_mat[tgt_idx - 1][(e_type - 1) * (1+object_num) + src_idx - 1] = 1
                #print s_o_id
                #print adj_mat



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
    data['adj_mat'] = adj_mat
    data['node_labels'] = node_labels
    data['relation_label'] = one_hot_relation_label
    data['graph_label'] = one_hot_graph_label
    data['sub_ind'] = -1


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
        #print type(sequence_id)
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
                current_segment_data['node_labels'] = next_segment_data['node_labels']
                data[sequence_id].append(current_segment_data)


    #  graph label
    subject_ids = dict()
    check_list = list()
    count = 0
    for annotation_path_file in os.listdir(paths.anno_root):
        subjects_path = os.path.join(paths.anno_root, annotation_path_file)
        subject_name = annotation_path_file.split('_')[0]
        print subject_name
        subject_index = metadata.subject_index[subject_name]
        #print subject_index
        subject_ids[subject_name] = list()
        num = 0
        for category in os.listdir(subjects_path):
            activitylabel = category
            one_hot_label = np.zeros((1, 10))
            graph_label = metadata.activity_index[activitylabel]

            #print graph_label
            #print one_hot_label.shape
            one_hot_label[0,graph_label] = 1

            #print "after equation"
            #print one_hot_label
            file = os.path.join(subjects_path,category,'activityLabel.txt')
            with open(file,'r') as f:
                lines=f.readlines()
                for line in lines:
                    count = count+1
                    sequence_id = line.split(',')[0]
                    #print sequence_id
                    subject_ids[subject_name].append(sequence_id)
                    check_list.append(sequence_id)
                    num = num +1

                #print sequence_id

                #print sequence_id
                #print len(data[sequence_id][:])
                    for i in range(len(data[sequence_id][:])):
                        #print 'lslslsl'
                        #print one_hot_label

                        data[sequence_id][i]['graph_label'] = one_hot_label
                        data[sequence_id][i]['sub_ind'] = subject_index
        print subject_name," has ",num," samples"

    print "%d samples",count
    pickle.dump(check_list, open(os.path.join(paths.tmp_root, 'check.p'), 'wb'))
    pickle.dump(data, open(os.path.join(paths.tmp_root, 'cad120_data_pred.p'), 'wb'))
    pickle.dump(subject_ids, open(os.path.join(paths.tmp_root, 'cad120_data_list.p'), 'wb'))

    # check which sample is not in subject
    for sequence_id in sequence_ids:
        if sequence_id not in check_list:
            print sequence_id

def main():
    paths = cad120_config.Paths()
    start_time = time.time()
    collect_data(paths)
    print('Time elapsed: {:.2f}s'.format(time.time() - start_time))


if __name__ == '__main__':
    main()
