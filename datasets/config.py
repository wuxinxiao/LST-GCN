"""
Created on Feb 17, 2017

@author: Siyuan Qi

Description of the file.

"""

import errno
import logging
import os

#import config


class Paths(object):
    def __init__(self):
        """
        Configuration of data paths
        member variables:
            data_root: The root folder of all the recorded data of events
            metadata_root: The root folder where the processed information (Skeleton and object features) is stored.
        """
        super(Paths, self).__init__()

        #self.detect_root1 = '/media/mcislab/new_disk/wangruiqi/data/something-detectron-result/thresh0.5/'
        self.o1 = '/media/mcislab/new_disk/wangruiqi/data/something-detectron-result/thresh0.5/'
        self.detect_root1 = '/media/mcislab/new_disk/wangruiqi/data/something-detectron-result/thresh0.5-edge/'
        self.detect_root2 = '/media/mcislab/new_disk/wangruiqi/data/something-detectron-result/thresh0.7/'
        self.detect_root3 = '/media/mcislab/new_disk/wangruiqi/data/something-detectron-result/nothresh/'
        self.feat_root= ''
        self.img_root = '/media/mcislab/new_disk/wangruiqi/data/something-something/20bn-something-something-v1/'
        
        self.detect_root_ucf = '/media/mcislab/new_disk/wangruiqi/data/UCF101-detectron-result/thresh0.5-edge/'
        self.detect_root_ucf_simplified = '/media/mcislab/new_disk/wangruiqi/data/UCF101-detectron-result/thresh0.5-edge_simplified/'
        self.detect_root_ucf_mmdet = '/media/mcislab/new_disk/wangruiqi/data/UCF101/UCF101_mmdet_pickle/' #ucf101 detect 10 frames per video
        self.detect_root_ucfall_mmdet = '/media/mcislab/new_disk/wangruiqi/data/UCF101/UCF101All_mmdet_pickle/' #ucf101 detect all frames per video
        #self.detect_root_bit_mmdet = '/media/mcislab/new_disk/wangruiqi/data/bit/BIT_mmdet_pickle/'
        self.detect_root_bit_mmdet = '/media/mcislab/new_disk/wangruiqi/data/bit/BIT_mmdet_pickle_allPerson/'
        self.detect_root_jhmdb_mmdet = '/media/mcislab/new_disk/wangruiqi/data/JHMDB/JHMDB_mmdet_pickle/'
        self.detect_root_hmdb = '/media/mcislab/new_disk/wangruiqi/data/HMDB51/hmdb51-detectron/'
        self.img_root_ucf = '/media/mcislab/new_disk/wangruiqi/data/UCF101/UCF101_rawimage/'
        self.img_root_jhmdb = '/media/mcislab/new_disk/wangruiqi/data/JHMDB/rgb_frms/'
        self.img_root_bit = '/media/mcislab/new_disk/wangruiqi/data/bit/frame/'
        self.img_root_hmdb = '/media/mcislab/new_disk/wangruiqi/data/HMDB51/frame/'
        self.rgb_resnext_ucf = '/media/mcislab/new_disk/wangruiqi/data/UCF101/3DResNext101Feat-V3/' #ucf101 3DResNext101
        self.rbg_resnext_ucf_more = '/media/mcislab/new_disk/wangruiqi/data/UCF101/3DResNext101Feat-V3.1/'
        self.avgrgb_resnext_ucf ='/media/mcislab/new_disk/wangruiqi/data/UCF101/3DResNext101_Feat_AVG/'
        self.rgb_bninc_ucf = '/media/mcislab/new_disk/wangruiqi/data/UCF101/BNINc_RGB_Feat_last2-V2/' #ucf101 BNInceptionV4-rgb
        self.rgb_res18_ucf = '/media/mcislab/new_disk/wangruiqi/data/UCF101/UCF101_Res18_Feat/' #ucf101 ResNet18-RGB
        self.rgb_res50_ucf = '/media/mcislab/new_ssd/wrq/data/UCF101/UCF101_Res50_Feat_imagenet/' #ucf101 ResNet50-RGB zhao added
        self.rgb_res50_ucf_fintuned = '/media/mcislab/new_ssd/wrq/data/UCF101/UCF101_Res50_Feat_imagenet_fintuned/'
        self.flow_res50_ucf = '/media/mcislab/new_ssd/wrq/data/UCF101/UCF101_Res50_Feat_flow/flow_images_res50/'
        self.flow_bninc_bit_v1 = '/media/mcislab/new_disk/wangruiqi/data/bit/BNINc_Flow_Feat_last2-V2/' #bit BNInceptionV4-flow last two layer,padding more 4 frames
        self.flow_bninc_bit_v2 = '/media/mcislab/new_disk/wangruiqi/data/bit/BNINc_Flow_Feat_last2-V2.1/' #bit BNInceptionV4-flow last two layer
        self.flow_res18_ucf = '/media/mcislab/new_disk/wangruiqi/data/UCF101/UCF101_Res18_Feat_flow/flow_images/'#ucf101 ResNet18-flow
        self.flow_bninc_ucf = '/media/mcislab/new_disk/wangruiqi/data/UCF101/BNINc_Flow_Feat_last2-V2/' #ucf101 BNInceptionV4-flow
        self.avgflow_bninc_ucf = '/media/mcislab/new_disk/wangruiqi/data/UCF101/BNINc_Flow_Feat_AVG/' #ucf101 BNInceptionV4-flow avg pooling for each ratio
        self.rgb_resnext_hmdb = '/media/mcislab/new_disk/wangruiqi/data/HMDB51/3DResNext101Feat/' #HMDB51 3DResNext101
        self.ap_resnet50_ucf = '/media/mcislab/new_ssd/wrq/data/UCF101/resnet50/'##prepared ResNet50 for UCF101 rgb != flow
        self.ap_resnet18_ucf = '/media/mcislab/new_disk/wangruiqi/data/UCF101/ap_resnet18/' # prepared ResNet18 for UCF101
        self.bninception_ucf = '/media/mcislab/tmp/wangruiqi/data/UCF101/ap_BNINc_sample10/' #prepared BNInception v4 feature for UCF101 new_disk->tmp
        self.bninception_ucf_similarity = '/media/mcislab/new_ssd/wrq/data/UCF101/inception_rgbflow_same_similiarity/'
        self.res18_ucf_similarity = '/media/mcislab/new_ssd/wrq/data/UCF101/res18_rgbflow_same_similiarity/'
        self.avgbninception_ucf = '/media/mcislab/new_disk/wangruiqi/data/UCF101/ap_BNINc_avg_sample10/'
        self.ap_avg3dresnext_ucf = '/media/mcislab/new_disk/wangruiqi/data/UCF101/ap_3DResNext101_avg_sample10/'#prepared 3DResNext101 for UCF101
        self.ap_3dresnext_ucf = '/media/mcislab/new_disk/wangruiqi/data/UCF101/ap_resnext3d_sample10/'
        self.resnet50_frame_feature = '/media/mcislab/new_ssd/wrq/data/UCF101/UCF101_Res50_Feat_imagenet/' #baseline1 zhao added it
        self.rgb_resnext_sample10_hmdb = '/media/mcislab/new_disk/wangruiqi/data/HMDB51/ap_sample10/'
        self.sthv2_final = '/media/mcislab/new_ssd/wrq/data/sth_v2/sth_v2_res18_final/'#res18 for somethingv2
        self.resnet50_ucf_rgbflow_same = '/media/mcislab/new_ssd/wrq/data/UCF101/resnet50_rgbflow_same/'#res50_for_ucf,rgb == flow,zhao addedd
        self.resnet50_ucf_rgbflow_same_fintuned = '/media/mcislab/new_ssd/wrq/data/UCF101/resnet50_rgbflow_same_fintuned/' #fintuned_res50_for_ucf
def set_logger(name='learner.log'):
    if not os.path.exists(os.path.dirname(name)):
        try:
            os.makedirs(os.path.dirname(name))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    logger = logging.getLogger(name)
    file_handler = logging.FileHandler(name, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s',
                                                "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)
    return logger
