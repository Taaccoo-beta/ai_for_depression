import torch.nn as nn
import torch.nn.functional as F
import torch


class HLSTM(nn.Module):
    def __init__(self,USE_GLOVE,pretrained_emb, token_size):
        super(HLSTM, self).__init__()

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

        self.layerNorm = nn.LayerNorm((300),eps=0.0001,elementwise_affine=True)
        
        self.linear = nn.Sequential(
            nn.Linear(300,125),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(125,32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32,1)
        )

    def forward(self,face_feat,voice_feat,text_feat):
        """
            face_feat: (b,300,204)
            voice_feat: (b,300,80)
            text_feat: (b,60)

        """

        T,_ = self.text_gru(self.embedding(text_feat))  #(b,60,300)
        V,_ = self.face_gru(face_feat)   #(b,300,300)
        A,_ = self.voice_gru(voice_feat) #(b,300,300)

        T = T[:,-1,:] #(b,300)
        V = V[:,-1,:]
        A = A[:,-1,:]
        
        
        output = T+V+A

        output = self.linear(output)

        return output

if __name__ == "__main__":
    net = HLSTM(0,0,7244)
    face_feat = torch.randn(64,300,204)
    voice_feat=torch.randn(64,300,80)
    text_feat=torch.ones((64,60),dtype=torch.long)

    out = net(face_feat,voice_feat,text_feat)

    
    print(out.shape)