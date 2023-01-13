## ref: https://blog.csdn.net/qq_36618444/article/details/106472126


# %% [markdown]
# ## 1. position embedding
# PE_{(pos, 2i)} = \sin(\frac{pos}{10000^{2i/d_{model}}})
# PE_{(pos, 2i+1)} = \cos(\frac{pos}{10000^{2i/d_{model}}})
# 


#%%
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConfigTrans(object):

    """配置参数"""
    def __init__(self):
        self.model_name = 'Transformer'
        # self.dropout = 0.5                              
        self.num_classes = 2                    # 类别数
        self.num_epochs = 200                # epoch数
        self.batch_size = 32             # mini-batch大小

        self.pad_size = 200                    # 短填长切


        self.learning_rate = 0.001                    # 学习率
        # self.embed = 100          # 字向量维度
        # self.dim_model = 100      # 需要与embed一样
        
        # self.hidden = 1024 
        # self.last_hidden = 512
        # self.num_head = 5       # 多头注意力，注意需要整除
        # self.num_encoder = 2    
        
config = ConfigTrans()




'''
params: embed-->word embedding dim      pad_size-->max_sequence_lenght
Input: x
Output: x + position_encoder
'''

class Positional_Encoding(nn.Module):

    def __init__(self, embed, pad_size, dropout):
        super(Positional_Encoding, self).__init__()

        # 单个x的embedding
        self.pe = torch.tensor(
            [
                [pos / (10000.0 ** (i // 2 * 2.0 / embed)) 
                 for i in range(embed)] 
                for pos in range(pad_size)]
            )
        # 偶数sin
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])  
        # 奇数cos
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])  
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
    	# 单词embedding与位置编码相加，这两个张量的shape一致
        out = x + nn.Parameter(self.pe, requires_grad=False).cuda()
        out = self.dropout(out)
        return out


#%% [markdown]
# multihead attention

# %%
'''
params: dim_model-->hidden dim      num_head
'''

class Scaled_Dot_Product_Attention(nn.Module):
    '''Scaled Dot-Product'''
    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None):
        attention = torch.matmul(Q, K.permute(0, 2, 1))  # Q*K^T
        if scale:
            attention = attention * scale
        # if mask:  # TODO change this
        #     attention = attention.masked_fill_(mask == 0, -1e9)
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)   
        return context


class Multi_Head_Attention(nn.Module):
    def __init__(self, dim_model, num_head, dropout=0.5):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0    # head数必须能够整除隐层大小
        self.dim_head = dim_model // self.num_head   # 按照head数量进行张量均分
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)  # Q，通过Linear实现张量之间的乘法，等同手动定义参数W与之相乘
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention() 
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)   # 自带的LayerNorm方法

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)  # reshape to batch*head*sequence_length*(embedding_dim//head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)
        # if mask:  # TODO
        #     mask = mask.repeat(self.num_head, 1, 1)  # TODO change this
        scale = K.size(-1) ** -0.5  # 根号dk分之一，对应Scaled操作
        context = self.attention(Q, K, V, scale) # Scaled_Dot_Product_Attention计算
        context = context.view(batch_size, -1, self.dim_head * self.num_head) # reshape 回原来的形状
        out = self.fc(context)   # 全连接
        out = self.dropout(out)
        out = out + x      # 残差连接,ADD
        out = self.layer_norm(out)  # 对应Norm
        return out



class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, dim_model, hidden, dropout=0.5):
        super(Position_wise_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(dim_model, hidden)
        self.fc2 = nn.Linear(hidden, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)   # 两层全连接
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out



#%% 
class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout):
        super(Encoder, self).__init__()
        self.attention = Multi_Head_Attention(dim_model, num_head, dropout)
        self.feed_forward = Position_wise_Feed_Forward(dim_model, hidden, dropout)

    def forward(self, x):
        out = self.attention(x)
        out = self.feed_forward(out)
        return out



class Transformer_cls(nn.Module):
    def __init__(self, 
                 pretrain_weights, 
                 num_class,
                 pad_size,
                 num_head=5, 
                 num_encoder=2,
                 embed_dim=100, hidden=1024,
                 dropout=0.5
                 ):
        super(Transformer_cls, self).__init__()

        # input text embedding 
        self.input_embedding = nn.Embedding.from_pretrained(pretrain_weights)
    

        # pos encoding
        self.postion_embedding = Positional_Encoding(embed_dim, pad_size, dropout=0.01)

        # encoders
        self.encoder = Encoder(embed_dim, num_head, hidden, dropout)
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(num_encoder)])

        # linear layer for classification
        self.fc1 = nn.Linear(pad_size * embed_dim, num_class)

        self.softmax = nn.Softmax()
        self.loss_ce = nn.CrossEntropyLoss()

    def forward(self, x, y):
        
        x = self.input_embedding(x)
        out = self.postion_embedding(x)
        for encoder in self.encoders:
            out = encoder(out)

        # 将三维张量reshape成二维，然后直接通过全连接层将高维数据映射为classes
        out = out.view(out.size(0), -1)
        # out = torch.mean(out, 1)    # 也可用池化来做，但是效果并不是很好
        out = self.fc1(out)
        
        pred  = torch.argmax(out, dim=-1)
        loss = self.loss_ce(out.view(-1, config.num_classes), y.view(-1))
        return loss, pred




if __name__ == "__main__":
    #%%
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    def get_position_encoding(seq_len, embed):
        pe = np.array([[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(seq_len)])  # 公式实现
        pe[:, 0::2] = np.sin(pe[:, 0::2])
        pe[:, 1::2] = np.cos(pe[:, 1::2])
        return pe


    pe = get_position_encoding(100,100)
    # print(pe[0])
    # print(pe[1])
    print(pe.size)
    sns.heatmap(pe)
    plt.xlabel('emb')
    plt.ylabel('seq_len')
    plt.show()
    # %%
