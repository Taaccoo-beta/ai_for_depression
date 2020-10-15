import torch.nn as nn
import torch.nn.functional as F
import torch



class CIM_Attention(nn.Module):
    def __init__(self):
        super(CIM_Attention,self).__init__()

    def forward(self,X,Y):
        """
         X--> (b,n1,300)
         Y--> (b,n2,300)
        """

        M1 = torch.matmul(X,Y.permute(0,2,1)) #(b,n1,n2) 
        M2 = torch.matmul(Y,X.permute(0,2,1)) #(b,n2,n1)

        N1 = torch.softmax(M1,dim=2) #(b,n1,n2)
        N2 = torch.softmax(M2,dim=2) #(b,n2,n1)

        O1 = torch.matmul(N1,Y) #(b,n1,300)
        O2 = torch.matmul(N2,X) #(b,n2,300)

        A1 = torch.mul(O1,X)
        A2 = torch.mul(O2,Y)
        return torch.cat((A1,A2),dim=1) # (b,n1+n2,300)

class EarlyFusionNet(nn.Module):

    def __init__(self,USE_GLOVE,pretrained_emb, token_size):
        super(EarlyFusionNet, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=300
        )

        # Loading the GloVe embedding weights
        if USE_GLOVE:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.face_gru = nn.LSTM(204,150,num_layers=1,bidirectional=True)  #b,n,300
        self.voice_gru = nn.LSTM(80,150,num_layers=1,bidirectional=True)  #b,n,300
        self.text_gru = nn.LSTM(300,150,num_layers=1,bidirectional=True)  #b,n,300 

        self.CIMA = CIM_Attention()

        self.transform = nn.Conv1d(300,1,kernel_size=5,stride=3)
        self.layerNorm = nn.LayerNorm((219),eps=0.0001,elementwise_affine=False)
        self.linear_layer = nn.Sequential(
            nn.Linear(219,100),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(100,20),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(20,1)
        )

    def forward(self,face_feat,voice_feat,text_feat):
        """
            face_feat: (b,200,204)
            voice_feat: (b,200,80)
            text_feat: (b,20)

        """
        T,_ = self.text_gru(self.embedding(text_feat))  #(b,20,300)
        V,_ = self.face_gru(face_feat)   #(b,100,300)
        A,_ = self.voice_gru(voice_feat) #(b,100,300)

        AttTV = self.CIMA(T,V) #(b,120,300)
        AttTA = self.CIMA(T,A) #(b,120,300)
        AttVA = self.CIMA(V,A) #(b,200,300)

        AttPren = torch.cat((AttTA,AttTV,AttVA,T,V,A),dim=1) #(b,100+100+20+120+120+200,300)
        out = self.transform(AttPren.permute(0,2,1))   #(b, 1,219)
        out = self.layerNorm(out.squeeze(1))
        out = self.linear_layer(out) #(b,2)

        return out  


if __name__ == "__main__":
    et = EarlyFusionNet(0)
    face_feat = torch.randn(64,200,204)
    voice_feat=torch.randn(64,200,80)
    text_feat=torch.ones((64,15),dtype=torch.long)

    out = net(face_feat,voice_feat,text_feat)
    print(out.shape)
    
