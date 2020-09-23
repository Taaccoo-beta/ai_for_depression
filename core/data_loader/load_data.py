# --------------------------------------------------------
# Produce training dataset
# Licensed under The MIT License [see LICENSE for details]
# Written by 
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.utils.data.dataset as Data  
import numpy as np 
from data_utils import tokenize,proc_face_voice_feature,proc_ques


class SplitedDataset(Data.Dataset):

    def __init__(self):
        
        self.face_feat_list = np.load("/mnt/sdc1/daicwoz/data_pro/split_feature_face.npy",allow_pickle=True)
        self.voice_feat_list = np.load("/mnt/sdc1/daicwoz/data_pro/split_feature_voice.npy",allow_pickle=True)
       
        self.attri_feat_list = np.load("/mnt/sdc1/daicwoz/data_pro/split_feature_0_4.npy",allow_pickle=True)

        self.text_feat_list = self.attri_feat_list[:,3]

        self.token_to_ix, self.pretrained_emb = tokenize(self.text_feat_list, True)
        self.token_size = self.token_to_ix.__len__()
        print('== Question token vocab size:', self.token_size)

        self.ans_to_ix = self.attri_feat_list[:,1]

    def __getitem__(self,idx):

        face_feat_iter = np.zeros(1)
        voice_feat_iter = np.zeros(1)
        text_feat_iter = np.zeros(1)

        text_iter = proc_ques(self.text_feat_list[idx],self.token_to_ix,15)
        

        face_feat_iter = self.face_feat_list[idx]
        voice_feat_iter = self.voice_feat_list[idx]
        
        ans_iter = self.ans_to_ix[idx]
        
        return torch.from_numpy(face_feat_iter), \
               torch.from_numpy(voice_feat_iter), \
               torch.from_numpy(text_iter), \
               torch.tensor(ans_iter,dtype=torch.long)

    def __len__(self):
        return  self.text_feat_list.__len__()

    
    def test(self):
        # print(self.face_feat_list.__len__())
        # print(self.voice_feat_list.__len__())
        # print(self.text_feat_list.__len__())
        pass 
        



if __name__ == "__main__":
    c = SplitedDataset()
    print(c[0])
    pass