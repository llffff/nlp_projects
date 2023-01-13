
import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalMaxPool1d(nn.Module):
    def __init__(self):
        super(GlobalMaxPool1d, self).__init__()

    def forward(self, x):
        # x shape: [batch_size, channel, seq_len]
        # return shape: [batch_size, channel, 1]
        # return F.max_pool1d(x, kernel_size=x.shape[2])

        x= x.squeeze(3)
        # print(x.shape)
        return F.max_pool1d(x, kernel_size=x.shape[2])



class TextCNN(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim, 
                 pretrain_weights,
                 kernel_sizes=[3,4,5],
                 num_channels=16,
                 num_class=2,
                 dropout1=0.5,
                 dropout2=0.8,
                 ):
        super(TextCNN, self).__init__()

        self.num_labels = num_class

        # learnable 
        self.embd = nn.Embedding.from_pretrained(pretrain_weights)
        
        # unlearnable
        self.constant_embd = nn.Embedding.from_pretrained(pretrain_weights)
        self.constant_embd.weight.requires_grad = False
        

        self.dropout_embd = nn.Dropout(dropout1)
        self.dropout_conv = nn.Dropout(dropout2)

        input_dim=1#2*embedding_dim
        
        
        self.conv11 = nn.Conv2d(
            in_channels=input_dim,
            out_channels=num_channels,
            kernel_size=(kernel_sizes[0], 2*embedding_dim))
        self.conv12 = nn.Conv2d(input_dim, num_channels, (kernel_sizes[1], 2*embedding_dim))
        self.conv13 = nn.Conv2d(input_dim, num_channels, (kernel_sizes[2], 2*embedding_dim))

        
        self.fc = nn.Linear(len(kernel_sizes) * num_channels, self.num_labels)

        
        self.pool = GlobalMaxPool1d()
        self.loss_ce= nn.CrossEntropyLoss()


    def forward(self, x, y):
        # print(self.constant_embd.weight.requires_grad)
        # print(self.constant_embd.weight.data[0])


        # input index:
        # print("x", x.shape)
        # (b, seq)

        # embedding:
        embeddings = torch.cat((
            self.embd(x),
            self.constant_embd(x)), dim=2) 
        embeddings = self.dropout_embd(embeddings)
        
        # print("embd:",embeddings.size())
        # [b, seqlen, 2*embd]
        # add dim:
        embeddings = embeddings.unsqueeze(1)
        # print("embd unsq1:", embeddings.size())
        # [b, 1, seqlen, 2*embd]
        
        
        # embeddings = embeddings.permute(0, 3, 1, 2)
        # embeddings = embeddings.permute(0, 2, 1)
        # print(embeddings.size())
        

        # in: 2*embd
        dim_ = 2
        x1_conv = self.conv11(embeddings)
        # print("x1_conv",x1_conv.shape) 
        x1 = self.pool(F.relu(x1_conv)).squeeze(dim_)        
        x2 = self.pool(F.relu(self.conv12(embeddings))).squeeze(dim_)
        x3 = self.pool(F.relu(self.conv13(embeddings))).squeeze(dim_)
        
        out = torch.cat((x1, x2, x3), dim=1)  # (batch, 3 * kernel_num)
        out = self.dropout_conv(out)

        
        outputs = self.fc(out)
        outputs = F.log_softmax(outputs, dim=1)

        loss = self.loss_ce(outputs.view(-1, self.num_labels), y.view(-1))
        pred  = torch.argmax(outputs, dim=-1)

        return loss,pred






if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    from torch.optim import Adam
    import torch
    import dataset_utils as data
    import utils as utils

    base_dir = "/home/mlsnrs/data/data/lff/ai-lab/lab3/"
    data_dir = base_dir + "../../data/"

    lr = 1e-3
    decay = 1e-4

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size= 64
    num_workers = 4
    num_epochs = 25 # 大约在n=10时 模型过拟合


    train_set = data.DatasetLoader(f"{data_dir}/fudan_nlp/train.data")
    dev_set = data.DatasetLoader(f"{data_dir}/fudan_nlp/valid.data")

    # building vocabulary
    vocab, idx_to_word = data.build_vocab([train_set,dev_set])

    # word->idx
    train_set.convert_word_to_idx(vocab)
    dev_set.convert_word_to_idx(vocab)

    

    # %%
    # pretrain words vector
    import torchtext.vocab as Vocab

    glove_vocab = Vocab.GloVe(name='6B', dim=100, cache=os.path.join(data_dir+"/fudan_nlp/", "glove"))
    print(len(glove_vocab.stoi)) # 400000
    print(glove_vocab[0].shape)

    vocab_weights = data.load_pretrained_embedding(vocab.keys(), glove_vocab)


    # exit(0)
    from textcnn import TextCNN

    vocab_weights = data.load_pretrained_embedding(vocab.keys(), glove_vocab)

    model =TextCNN(
        vocab_size=len(vocab),                
        embedding_dim=100, 
        pretrain_weights=vocab_weights,
        kernel_sizes=[3,4,5], num_channels=16,
        num_class=2,dropout2=0.8
        )

    print(model)



    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=decay) 


    # train
    model.train()        
    
    tr_loss = 0
    tr_pred, tr_y = [],[]
    
    train_data_iter=train_set.get_random_iter(num_workers=num_workers, batch_size=batch_size)
    for step, batch in enumerate((train_data_iter)):
        # forward
        x,y=batch
        loss, pred = model(x.to(device), y.to(device))

        # backward
        loss.backward()
        optimizer.step()
        model.zero_grad()


        print(f"loss:{loss}, \npred:{pred},\ny:{y}")
        break

    
