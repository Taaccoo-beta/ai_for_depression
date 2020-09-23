import numpy as np 
import pandas as pd 
import os 
from collections import Counter

def get_padding_threshold(data_list):
    average = int(np.mean(data_list))
    for i in range(average,np.max(data_list)):
        count = 0 
        for item in data_list:
            if item <= i:
                count+=1
        if count/data_list.__len__()>=0.9:
            return i 

def analysis_sentence_length(root_dir):
    for item in range(300,493):
        alignment_feature_path = os.path.join(root_dir,"alignment/{}_alignment_feature.csv".format(item))
        try:
            f = pd.read_csv(alignment_feature_path,sep=",")
        except:
            continue
        count = 0 
        value_list = []
        for ii in f["value"]:
            ii.split(" ")
            try:
                if ii.split(" ").__len__()<15:
                    count+=1 
                value_list.append(ii.split(" ").__len__())
            except:
                print("delete",ii,item,value_list.__len__())
        print(item,count/value_list.__len__())



if __name__ == "__main__":
    data_face = np.load("/mnt/sdc1/daicwoz/data_pro/split_feature_voice.npy",allow_pickle=True).tolist()