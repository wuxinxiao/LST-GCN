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


        self.tmp_root = '/media/mcislab/new_disk/wangruiqi/data/CAD120/ap-feature/'
        self.abl_root = '/media/mcislab/new_disk/wangruiqi/data/CAD120/ap-feature-ggnn/' #for GGNN
        #self.tmp_root = '/media/mcislab/wrq/CAD120/reco-feature/'
        self.anno_root = '/media/mcislab/new_disk/wangruiqi/data/CAD120/annotations/'
        self.data_root = '/media/mcislab/new_disk/wangruiqi/data/CAD120/'




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
