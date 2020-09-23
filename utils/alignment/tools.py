import os 
import numpy as np
import pandas as pd 
import torch

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
    duration_save = "/mnt/sdc1/daicwoz/audio_duration_dict_ms.npy"
    np.save(duration_save,duration_dict)


class ProductAlignment:
    def __init__(self,root_path="/mnt/sdc1/daicwoz"):
        self.path_Transcript_format = None
        self.path_face3d_format = None
        self.path_voice_feature_format = None
        self.save_csv_filename = None
        self.path_duration = os.path.join(root_path,"audio_duration_dict_ms.npy")

    def generate_path(self,item):
        self.path_Transcript_format = os.path.join(root_path,"{}_P/{}_TRANSCRIPT.csv".format(item,item))
        self.path_face3d_format = os.path.join(root_path,"{}_CLNF_features3D.txt".format(item,item))
        self.path_voice_feature_format = os.path.join(root_path,"{}_P/{}_encoded_AUDIO.pt".format(item,item))
        self.save_csv_filename = os.path.join(root_path,"alignment/{}_alignment_feature.csv".format(item,item))

    def load_duration(self,file_name):
        f = np.load(file_name,allow_pickle=True).tolist()
        return f 

    def get_voice_feature_len(self,file_name):
        f = torch.load(path_voice_feature)
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

        return start_id,stop_id

    def generate_csv(self):
        total_time = load_duration(path_duration)
        for item in range(300,493):
            self.generate_path(item)
            
            try:
                f = pd.read_csv(self.path_Transcript_format,sep="\t")
                item_total_time = total_time[item]
                print("totoal time:",item,item_total_time)

                face_feature_len = self.get_feature_3d_len(self.path_face3d_format)
                
                voice_feature_len = self.get_voice_feature_len(self.path_voice_feature_format)
            except:
                continue 

            list_face_id=[]
            list_voice_id=[]
            for item in range(f.shape[0]):
                start_t = int(f["start_time"].loc[item]*1000)
                stop_t = int(f["stop_time"].loc[item]*1000)
            
                id_start_face,id_end_face = self.calc_feature_location(start_t,stop_t,item_total_time,face_feather_len)
                list_face_id.append([id_start_face,id_end_face])
            
                id_start_voice,id_end_voice = self.calc_feature_location(start_t,stop_t,item_total_time,voice_feature_len)
                list_voice_id.append([id_start_voice,id_end_voice])
                list_face_id = np.array(list_voice_id)
                list_voice_id = np.array(list_voice_id)

                f["face_id_start"] = list_face_id[:,0]
                f["face_id_end"] = list_face_id[:,1]

                f["voice_id_start"] = list_voice_id[:,0]
                f["voice_id_end"] = list_voice_id[:,1]
                print(save_csv_filename)
                f = f.dropna(how="any")
                f.to_csv(save_csv_filename,index=0)
    



if __name__ == "__main__":
    pass 