import os 
import numpy as np
import pandas as pd 
import torch
import re

def get_duraion():

    data_list = []
    for item in range(300,493):
        path = '/mnt/sdc1/daicwoz/'+str(num)+'_P/'+str(num)+'_AUDIO.wav'
        try:
            duration = librosa.get_duration(filename=path)
            ms = duration*1000 
            data_list.append([item,int(ms)])
        except:
            pass 
    duration_dict = {item[0]:item[1] for item in data_list}
    duration_save = "/mnt/sdc1/daicwoz/data_pro/audio_duration_dict_ms.npy"
    np.save(duration_save,duration_dict)


class ProductAlignment:
    """
    Produce alignment feature 

    """

    def __init__(self,root_path="/mnt/sdc1/daicwoz"):
        self.path_Transcript_format = None
        self.path_face3d_format = None
        self.path_voice_feature_format = None
        self.save_csv_filename = None

        self.root_path = root_path
        self.path_duration = os.path.join(self.root_path,"data_pro/audio_duration_dict_ms.npy")

    def generate_path(self,item):
        self.path_Transcript_format = os.path.join(self.root_path,"{}_P/{}_TRANSCRIPT.csv".format(item,item))
        self.path_face3d_format = os.path.join(self.root_path,"{}_P/{}_CLNF_features3D.txt".format(item,item))
        self.path_voice_feature_format = os.path.join(self.root_path,"{}_P/{}_encoded_AUDIO.pt".format(item,item))
        self.save_csv_filename = os.path.join(self.root_path,"data_pro/alignment/{}_alignment_feature.csv".format(item,item))
        self.save_csv_concat_filename = os.path.join(self.root_path,"data_pro/alignment/{}_alignment_concat_feature.csv".format(item,item))

    def load_duration(self,file_name):
        f = np.load(file_name,allow_pickle=True).tolist()
        return f 

    def get_voice_feature_len(self,file_name):
        f = torch.load(file_name)
        return f.shape[0]

    def get_feature_3d_len(self,file_name):
        with open(file_name,"r") as f:
            feature_list = f.readlines()
            return feature_list.__len__()-1

    def calc_feature_location(self,start_t,stop_t,total_time,feature_num):
        """
            returns: id_start,id_end 
        """
        start_id = start_t/total_time*feature_num 
        stop_id = stop_t/total_time*feature_num 

        return int(np.round(start_id)),int(np.round(stop_id))


    def get_item_from_index(self,start_ellie,end_participant,file_dataframe):
        start_time = file_dataframe.iloc[start_ellie+1,0]  # participant start time
        stop_time = file_dataframe.iloc[end_participant,1] # participant stop time 
        speaker = "Participant" 
        
        value = ""                                         # all of participant 
        for i in range(start_ellie+1,end_participant+1):
            value+=" "+file_dataframe.iloc[i,3]
        
        face_id_start = file_dataframe.iloc[start_ellie+1,4]  # face_id_start for participant during speaking
        face_id_end = file_dataframe.iloc[end_participant,5]

        voice_id_start = file_dataframe.iloc[start_ellie+1,6]
        voice_id_end = file_dataframe.iloc[end_participant,7]

        question = file_dataframe.iloc[start_ellie,3]  # question asked by Ellie
        
        new_item = pd.Series({"start_time":start_time,"stop_time":stop_time,"speaker":speaker,"value":value,"face_id_start":face_id_start,"face_id_end":face_id_end,"voice_id_start":voice_id_start,"voice_id_end":voice_id_end,"question":question})
        return new_item

    def csv_filter(self,f_dataframe):
        index = 0 
        length = f_dataframe.__len__() 
        patient_part = f_dataframe["speaker"]
        print(length)

        start_ellie = 0
        end_participant = 0
        new_csv_dataframe = pd.DataFrame()

        while index<length:
            try:
                if patient_part[index] == "Ellie" and patient_part[index+1]=="Participant":
                    i = 0
                    while patient_part[index+i+1] == "Participant":
                        i = i + 1
                    start_ellie = index
                    end_participant = index+i

                    new_item = get_item_from_index(start_ellie,end_participant,f_dataframe)
                    new_csv_dataframe.append(new_item)
                    print(start_ellie,end_participant)
                    index = index+i+1
            except:
                pass   # index out of length
            else:
                index = index+1
        return new_csv_dataframe

    def generate_csv(self,if_concat):
        """
            return:
                (start_time	stop_time	speaker	value	face_id_start	face_id_end	voice_id_start	voice_id_end)
        """
        total_time = self.load_duration(self.path_duration)
        count = 0 
        for item in range(300,493):
            self.generate_path(item)
            try:
                f = pd.read_csv(self.path_Transcript_format,sep="\t")
                item_total_time = total_time[item]
                print("totoal time:",item,item_total_time)

                face_feature_len = self.get_feature_3d_len(self.path_face3d_format)
                
                voice_feature_len = self.get_voice_feature_len(self.path_voice_feature_format)
                count+=1
            except:
                print("except:",item)
                continue 

            list_face_id=[]
            list_voice_id=[]
            for item in range(f.shape[0]):
                start_t = int(f["start_time"].loc[item]*1000)
                stop_t = int(f["stop_time"].loc[item]*1000)
            
                id_start_face,id_end_face = self.calc_feature_location(start_t,stop_t,item_total_time,face_feature_len)
                list_face_id.append([id_start_face,id_end_face])
            
                id_start_voice,id_end_voice = self.calc_feature_location(start_t,stop_t,item_total_time,voice_feature_len)
                list_voice_id.append([id_start_voice,id_end_voice])
                
            list_face_id = np.array(list_face_id)
            list_voice_id = np.array(list_voice_id)

            f["face_id_start"] = list_face_id[:,0]
            f["face_id_end"] = list_face_id[:,1]

            f["voice_id_start"] = list_voice_id[:,0]
            f["voice_id_end"] = list_voice_id[:,1]
            print(self.save_csv_filename)
            f = f.dropna(how="any")


            if if_concat:
                patient_part = f[f["speaker"]=="Participant"]
                index_ellie = patient_part.index-1
                data_question = f.iloc[index_ellie,3].tolist()
                
                ## filter () 
                p2 = re.compile(r'[(](.*)[)]', re.S)
                data_pro = []
                for item in data_question:
                    s = re.findall(p2,item)
                    if s != []:
                        data_pro.append(s[0])
                    else:
                        data_pro.append(item)
                patient_part["question"] = data_pro

                patient_part.to_csv(self.save_csv_concat_filename,index=0)
            else:
                f.to_csv(self.save_csv_filename,index=0)
                
    



if __name__ == "__main__":
    P = ProductAlignment()
    P.generate_csv(True) 