# --------------------------------------------------------
# Basic setting for path
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os 

class PATH:
    def __init__(self):

        
        self.DATASET_PATH = '/mnt/'

        self.FEATURE_PATH = ''

        self.init_path()


    def init_path(self):

        self.TEXT_FEAT_PATH = self.FEATURE_PATH + 'text_feature'

        self.FACE_FEAT_PATH = {
            'face2d': self.DATASET_PATH + 'v2_OpenEnded_mscoco_train2014_questions.json',
            'face3d': self.DATASET_PATH + 'v2_OpenEnded_mscoco_val2014_questions.json',
        }

        self.AUDIO_FEAT_PATH = self.FEATURE_PATH + 'audio_feature'

        self.RESULT_PATH = './results/result_test/'
        self.PRED_PATH = './results/pred/'
        self.LOG_PATH = './results/log/'
        self.CKPTS_PATH = './ckpts/'

        if 'result_test' not in os.listdir('./results'):
            os.mkdir('./results/result_test')

        if 'pred' not in os.listdir('./results'):
            os.mkdir('./results/pred')


        if 'log' not in os.listdir('./results'):
            os.mkdir('./results/log')

        if 'ckpts' not in os.listdir('./'):
            os.mkdir('./ckpts')


    def check_path(self):
        print('Checking dataset ...')

        """"""
        print('Finished')
        print('')

