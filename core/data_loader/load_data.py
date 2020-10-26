# --------------------------------------------------------
# Produce training dataset
# Licensed under The MIT License [see LICENSE for details]
# Written by 
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.utils.data.dataset as Data  
import numpy as np 
import sys
sys.path.append("..")
from data_utils import *


#[item,item_label,segment_question,segment_answer,segment_face,segment_voice]

class SplitedDataset(Data.Dataset):

    def __init__(self,train_or_test):
        self.train_or_test = train_or_test 
        # data_train_index = np.load("/mnt/sdc1/daicwoz/data_pro/participant_index_train.npy").tolist()
        # data_test_index = np.load("/mnt/sdc1/daicwoz/data_pro/participant_index_test.npy").tolist()

    
        if self.train_or_test:
            self.face_feat_list = np.load("/mnt/sdc1/daicwoz/data_pro/split_feature_face.npy",allow_pickle=True)
            
            self.voice_feat_list = np.load("/mnt/sdc1/daicwoz/data_pro/split_feature_voice.npy",allow_pickle=True)
            
        else:
            self.face_feat_list = np.load("/mnt/sdc1/daicwoz/data_pro/split_feature_face_test.npy",allow_pickle=True)
            self.voice_feat_list = np.load("/mnt/sdc1/daicwoz/data_pro/split_feature_voice_test.npy",allow_pickle=True)
            
        
        self.text_feat_train_list = np.load("/mnt/sdc1/daicwoz/data_pro/split_feature_0_4.npy",allow_pickle=True)
        self.text_feat_test_list = np.load("/mnt/sdc1/daicwoz/data_pro/split_feature_0_4_test.npy",allow_pickle=True)
        
        self.attri_feat_list = self.text_feat_train_list.tolist()+self.text_feat_test_list.tolist()
        self.attri_feat_list = np.array(self.attri_feat_list)
        self.attri_feat_list = np.concatenate((self.attri_feat_list[:,2],self.attri_feat_list[:,3]),axis=0)
        self.token_to_ix, self.pretrained_emb = tokenize(self.attri_feat_list, True)
        self.token_size = self.token_to_ix.__len__()
        print('== Question token vocab size:', self.token_size)

        if self.train_or_test:
            self.data_text = self.text_feat_train_list[:,2:4]  # get question and answer
            self.ans_to_ix = self.text_feat_train_list[:,1]
        else:
            self.data_text = self.text_feat_test_list[:,2:4]   # get question and answer
            self.ans_to_ix = self.text_feat_test_list[:,1]

        

    def __getitem__(self,idx):

        face_feat_iter = np.zeros(1)
        voice_feat_iter = np.zeros(1)
        text_feat_iter = np.zeros(1)

        text_iter = proc_ques(self.data_text[idx],self.token_to_ix,14,45)
        

        face_feat_iter = self.face_feat_list[idx]
        voice_feat_iter = self.voice_feat_list[idx]
        # print("face:",self.face_feat_list.__len__())
        ans_iter = self.ans_to_ix[idx]
        
        return torch.from_numpy(face_feat_iter).type(torch.float), \
               torch.from_numpy(voice_feat_iter).type(torch.float), \
               torch.from_numpy(text_iter).type(torch.long), \
               torch.tensor(ans_iter,dtype=torch.long)

    def __len__(self):
        if self.train_or_test:
            return self.text_feat_train_list.__len__()
        else:
            return self.text_feat_test_list.__len__()
       

    
    def test(self):
        # print(self.face_feat_list.__len__())
        # print(self.voice_feat_list.__len__())
        # print(self.text_feat_list.__len__())
        pass 
        



if __name__ == "__main__":
    c = SplitedDataset(0)
    print(c[0][1].shape)
    print(c[0][2].shape)
    pass