
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# attention部分参考: https://blog.csdn.net/qq_36618444/article/details/106472126

class LSTM_Attention(nn.Module):
    def __init__(self,vocab_size,
                 embedding_dim,hidden_dim,
                 n_layers,num_class, 
                 pretrain_weights,dropout=0.5):
        super(LSTM_Attention,self).__init__()

        self.num_class = num_class
        
        self.embedding = nn.Embedding.from_pretrained(pretrain_weights)
        self.constant_embedding = nn.Embedding.from_pretrained(pretrain_weights)
        self.constant_embedding.weight.requires_grad=False
        
        
        self.rnn = nn.LSTM(
            input_size = 2*embedding_dim,
            hidden_size = hidden_dim,
            num_layers = n_layers,
            batch_first = True, 
            bidirectional=True
            )

        # 用output分类
        # dim = hidden_dim*2
        
        # 用hidden分类
        dim = hidden_dim
        
        self.W_Q = nn.Linear(dim,dim,bias =False)
        self.W_K = nn.Linear(dim,dim,bias =False)
        self.W_V = nn.Linear(dim,dim,bias =False)

        # 类似多头注意力 linear+残差
        # self.att_fc = nn.Linear(dim, dim)
                
        # 分类fc
        self.fc = nn.Linear(dim,num_class)
        #dropout
        self.dropout = nn.Dropout(dropout)

        self.loss_ce = nn.CrossEntropyLoss()
        # self.softmax = nn.Softmax()
        
    
    def attention(self,Q,K,V):
        '''
        Q,K,V
        '''
        # n
        d_k = K.size(-1) # dk=dq

        # QK^T/dK: b, nxn
        att_score = torch.matmul(Q,K.transpose(1,2)) / math.sqrt(d_k)
        # softmax(·): b, nxn
        att_norm = F.softmax(att_score,dim=-1)

        # att·V: b, nxn
        att_map = torch.matmul(att_norm,V)

        # att·V: b, n
        # output = att_map.sum(1)
        return att_map,att_norm



    def forward(self,x,y):       
        # embeddings = self.embedding(x)
        embeddings = torch.cat((
            self.embedding(x),
            self.constant_embedding(x)), dim=2) 
        
        embeddings = self.dropout(embeddings)    
        # print('x:', x.size()) 
        # # x: torch.Size([64, seqlen])
        # print('e:', embedding.size())
        # # e: torch.Size([64, seqlen, 100])

        
        output,(h_n,c) = self.rnn(embeddings)  

        # attention
        # 用 output 表示
        # att_in = output 
        # 用 hidden state 表示
        att_in = h_n.transpose(0,1)

        # print("att_in", att_in.shape)
        # att_in torch.Size([batch, num_layers*2, hidden_num])
        

        Q = self.W_Q(att_in)        
        K = self.W_K(att_in)
        V = self.W_V(att_in)
        
        att_map,att_norm = self.attention(Q,K,V)     
        # ffn?
        # att_fc = self.att_fc(att_map)
        att_fc = att_map
        att_out = att_fc.sum(1)
        
        # out = out.view(x.size()[0], -1, att_map.size()[-1])
        # out = out + att_in 
        # out = att_map

        output = self.dropout(att_out)    
        # fc
        out = self.fc(att_out)

        pred  = torch.argmax(out, dim=-1)
        loss = self.loss_ce(out.view(-1, self.num_class), y.view(-1))    
        return loss, pred

'''
x: torch.Size([64, 47])
e: torch.Size([64, 47, 100])
output: torch.Size([64, 47, 512])
h: torch.Size([4, 64, 256])
c: torch.Size([4, 64, 256])
att_in: torch.Size([64, 47, 512])
Q: torch.Size([64, 47, 512])
att_map: torch.Size([64, 47, 512])
att_norm: torch.Size([64, 47, 47])
att_fc: torch.Size([64, 47, 512])
out sum: torch.Size([64, 2])
out: torch.Size([64, 47, 2])
'''



if __name__ == "__main__":


    # test forward once
        
    base_dir = "/home/mlsnrs/data/data/lff/ai-lab/lab3/"
    data_dir = base_dir + "../../data/"
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size= 64
    num_workers = 4

    import dataset_utils as data
    train_set = data.DatasetLoader(f"{data_dir}/fudan_nlp/train.data")
    vocab, idx_to_word = data.build_vocab([train_set])

    # word->idx
    train_set.convert_word_to_idx(vocab)

    import torchtext.vocab as Vocab
    import os
    glove_vocab = Vocab.GloVe(name='6B', dim=100, cache=os.path.join(data_dir+"/fudan_nlp/", "glove"))
    print(len(glove_vocab.stoi)) # 400000
    print(glove_vocab[0].shape)

    vocab_weights = data.load_pretrained_embedding(vocab.keys(), glove_vocab)

    model =LSTM_Attention(
    vocab_size=len(vocab),                
    embedding_dim=100, 
    hidden_dim=256, n_layers=2, num_class=2,
    pretrain_weights=vocab_weights)

    
    print(model)
    model.to(device)
    from torch.optim import Adam
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    for _batch in train_set.get_random_iter(batch_size, num_workers):
        model.eval()
        x,y=_batch
        loss, pred = model(x.to(device), y.to(device))
        break
