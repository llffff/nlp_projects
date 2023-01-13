import torch
import torch.nn as nn
import numpy as np


def init_liner(input_linear, seed = 1337):
    """initiate weights in linear layer
    """
    torch.manual_seed(seed)
    scope = np.sqrt(6.0 / (input_linear.weight.size(0)+input_linear.weight.size(1)))
    nn.init.uniform_(input_linear.weight,- scope, scope)
    if input_linear.bias is not None:
        input_linear.bias.data.zero_()


class LSTMModel(nn.Module):
    def __init__(self, vocab_size ,
                 embedding_dim, 
                 hidden_dim,
                 pretrain_weights,
                 n_layers=2,
                 num_class=2,
                 dropout=0.5):

        super(LSTMModel, self).__init__()


        self.num_class = num_class
        self.hidden_dim= hidden_dim

        self.dropout = nn.Dropout(dropout)

        # self.embd =  nn.Embedding(vocab_size, embedding_dim)
        self.embd =  nn.Embedding.from_pretrained(pretrain_weights)
        self.constant_embd =  nn.Embedding.from_pretrained(pretrain_weights)
        self.constant_embd.weight.requires_grad= False
        
        self.lstm = nn.LSTM(input_size=embedding_dim*2, 
                                hidden_size= hidden_dim,
                                num_layers=n_layers, 
                                batch_first = True, bidirectional=True)

        self.out = nn.Linear(hidden_dim*2, num_class)
        init_liner(self.out)

        self.loss_ce= nn.CrossEntropyLoss()

    def forward(self,x,y):        
        
  
        batch_size, seq_len = x.size()

        embd = self.embd(x)
        cons_embd= self.constant_embd(x)
        # torch.Size([32, 36, 100])

        embd = torch.cat((embd, cons_embd), dim=2)
        # torch.Size([32, 36, 200])

        
        x = self.dropout(embd)

        rnn_out,_ =self.lstm(x)
        # torch.Size([32, 36, 400])
        
        rnn_out = rnn_out.view(batch_size, seq_len, 2, self.hidden_dim)
        # print(rnn_out.shape)
        # torch.Size([32, 36, 2, 200])
        
        rnn_out = torch.cat([rnn_out[:,-1,0,:], rnn_out[:,0,1,:]], dim=-1)
        rnn_out = self.dropout(rnn_out)
        logits = self.out(rnn_out)

        
        loss = self.loss_ce(logits.view(-1, self.num_class), y.view(-1))
     
        pred  = torch.argmax(logits, dim=-1)

        return loss, pred
 
