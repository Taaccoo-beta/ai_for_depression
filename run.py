import torch
import torch.nn as nn 
import torch.optim as optimizer 
from torch.utils.data import DataLoader,Subset

import sys
sys.path.append("core/data_loader")

from core.models.EarlyFusion import *
from core.data_loader.load_data import *


def get_accuracy(result_list,test_dataset):
    ans_to_ix = test_dataset.ans_to_ix 
    result_list = np.array(result_list)

    print(ans_to_ix.shape)
    print(result_list.shape)
    return (ans_to_ix == result_list).sum() / ans_to_ix.__len__() 






def run():
    print("loading dataset ....")
    train_dataset = SplitedDataset(train_or_test=True)  #25499
    test_dataset = SplitedDataset(train_or_test=False)  #8946
   
    

    print("train_data_len:",train_dataset)
    print("test_data_len:",test_dataset) 


    total_number_of_test_dataset = test_dataset.__len__()
    train_data_iter = DataLoader(train_dataset,batch_size=128,shuffle=True) #
    test_data_iter = DataLoader(test_dataset,batch_size=128,shuffle=False)





    device = torch.device('cuda:1')

    net = EarlyFusionNet(True,train_dataset.pretrained_emb,train_dataset.token_size).to(device)


    
    criteon = nn.BCEWithLogitsLoss()

    
    opt = optimizer.Adam(net.parameters(), lr=0.001,weight_decay=0.001)
   
    print(net)
    count = 0
    for epoch in range(100):

        net.train()

        
        for batch_idx,(face,voice,text,ans) in enumerate(train_data_iter):
            face,voice,text,ans = face.to(device),voice.to(device),text.to(device),ans.to(device)
            
            out = net(face,voice,text).squeeze(1)
            loss = criteon(out,ans)

            opt.zero_grad()
            loss.backward()
            opt.step()
            
        
            if batch_idx%20 ==0:
                print("epoch: {}, loss: {}".format(epoch,loss))
    #     print(epoch, loss.item())
    #     pred = sm(logits)
    #     #print("pred: ", pred, "label: ", label)
        net.eval()
        with torch.no_grad():
            
            total_correct = 0
            total_num = 0
            total_positive = 0
            total_true = 0
            total_true_positive = 0
            result_list = []
            co = 0
            total_loss = 0

            for face_test,voice_test,text_test,ans_test in test_data_iter:
                face_test,voice_test,text_test,ans_test = face_test.to(device),voice_test.to(device), \
                                                          text_test.to(device),ans_test.to(device)
                result = net(face_test,voice_test,text_test)
                lo = criteon(result.squeeze(1),ans_test)
                total_loss+=lo
                
                result = torch.gt(torch.sigmoid(result),0.5).int()                
                
                result_list.extend(result.cpu().squeeze(1).numpy().tolist())
                
            total_loss = total_loss
            print("test_loss:",total_loss)
            print("accuracy:",get_accuracy(result_list,test_dataset))
        count+=1 
        print("epoch:",count)
        
                
            


                

           

if __name__ == "__main__":
    run()


