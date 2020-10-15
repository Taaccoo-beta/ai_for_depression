import numpy as np 
import pandas as pd 
import librosa
import torch 
import sys

sys.path.append('../../')
from core.data_loader.data_utils import proc_face_voice_feature

def normalize_features(data_list):    
    mean = np.mean(data_list,axis=0,keepdims=True)
    dev = np.std(data_list,axis=0,keepdims=True)+0.001
    data_list = (data_list-mean)/dev
    return data_list



def produce_splited_feature(pad_size,isTrain):
    if isTrain:
        labels = np.load("/mnt/sdc1/daicwoz/data_pro/train_title2label.npy",allow_pickle=True).tolist()
    else: 
        labels = np.load("/mnt/sdc1/daicwoz/data_pro/test_title2label.npy",allow_pickle=True).tolist()

    print("Length of lgit abels.keys():",labels.keys().__len__())
    data_list = []
    for item in labels.keys():
        path_Transcript_format = "/mnt/sdc1/daicwoz/data_pro/alignment/{}_alignment_feature.csv".format(item,item)
        path_face3d_format = "/mnt/sdc1/daicwoz/{}_P/{}_CLNF_features3D.txt".format(item,item)
        path_voice_feature_format="/mnt/sdc1/daicwoz/{}_P/{}_encoded_AUDIO.pt".format(item,item)

        
        try:
            f = pd.read_csv(path_Transcript_format,sep=",")
            # start_time	stop_time	speaker	value	face_id_start	face_id_end	voice_id_start	voice_id_end
           
            face_feature = open(path_face3d_format,'r').readlines()[1:]
            face_feature = [list(map(float,face_feature[i].split(",")))[4:] for i in range(face_feature.__len__())]
            face_feature = normalize_features(face_feature)
            face_feature = np.array(face_feature)
            
            voice_feature = torch.load(path_voice_feature_format).numpy()
            voice_feature = normalize_features(voice_feature)
            
            item_label = labels[item]
        except:
            continue
        
        
        for i in range(f.__len__()):
            segment_speak = f.iloc[i][2]
            segment_sentence = f.iloc[i][3]
            segment_face = face_feature[f.iloc[i][4]:f.iloc[i][5]]
            segment_voice = voice_feature[f.iloc[i][6]:f.iloc[i][7]]

            segment_face = proc_face_voice_feature(segment_face,pad_size)
            segment_voice = proc_face_voice_feature(segment_voice,pad_size)

            data_list.append([item,item_label,segment_speak,segment_sentence,segment_face,segment_voice])
            
           
        
        print(item)
        
    data_a = np.array(data_list)

    if isTrain:
        np.save("/mnt/sdc1/daicwoz/data_pro/split_feature_0_4.npy",data_a[:,0:4])
        np.save("/mnt/sdc1/daicwoz/data_pro/split_feature_face.npy",data_a[:,4])
        np.save("/mnt/sdc1/daicwoz/data_pro/split_feature_voice.npy",data_a[:,5])
    else: 
        np.save("/mnt/sdc1/daicwoz/data_pro/split_feature_0_4_test.npy",data_a[:,0:4])
        np.save("/mnt/sdc1/daicwoz/data_pro/split_feature_face_test.npy",data_a[:,4])
        np.save("/mnt/sdc1/daicwoz/data_pro/split_feature_voice_test.npy",data_a[:,5])


def get_participant_index():
    text_feat_train_list = np.load("/mnt/sdc1/daicwoz/data_pro/split_feature_0_4.npy",allow_pickle=True)
    text_feat_test_list = np.load("/mnt/sdc1/daicwoz/data_pro/split_feature_0_4_test.npy",allow_pickle=True)
    data_train = pd.DataFrame(text_feat_train_list.tolist())
    data_test = pd.DataFrame(text_feat_test_list.tolist())

    data_train_index = data_train[data_train[2]=="Participant"].index.tolist()
    data_test_index = data_test[data_test[2]=="Participant"].index.tolist()

    np.save("/mnt/sdc1/daicwoz/data_pro/participant_index_train.npy",data_train_index)
    np.save("/mnt/sdc1/daicwoz/data_pro/participant_index_test.npy",data_test_index)

    print("done")

if __name__ == "__main__":
    produce_splited_feature(100,False)
    #get_participant_index()