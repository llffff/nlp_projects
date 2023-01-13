#%%
import pickle
import time
import numpy as np
import torch
import torch.nn as nn
import torch.onnx

from model import BiLSTM_CRF
from epoch import train, evaluate


#%%
#parser
import argparse
parser = argparse.ArgumentParser(description='BiLSTM-CNN-CRF')


# Default settings
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
parser.add_argument('--device', type=str, default='cuda:0',
                    help='device for computing')
parser.add_argument('--path_data', type=str, default='./data/raw/',
                    help='path of the data corpus')
parser.add_argument('--path_processed', type=str, default='./data/data_bundle.pkl',
                    help='path of the processed data information')
parser.add_argument('--path_filtered', type=str, default='./data/data_filtered_bundle.pkl',
                    help='path to save the filtered processed data')
parser.add_argument('--path_pretrained', type=str, default='./data/trained-model-cpu',
                    help='path of the data corpus')
parser.add_argument('--path_model', type=str, default='./result/models/model.pt',
                    help='path of the trained model')


import sys
sys.argv=['main.py']
args = parser.parse_args()
args.device = torch.device(args.device)


# 训练参数
batch_size=128
epochs=20 # 1 for test
es_patience_max=10 # max early stopped patience
num_worker=5



## 输入预处理
digits_zero = True
args.is_lowercase = True  
args.START_TAG, args.STOP_TAG = '<START>','<STOP>'
# 数字处理为0(digits)
# 将word转换为小写(char level不变)
# 补开头和结尾标记


## 标签预处理
args.tag_scheme = 'BIO'
# 将I-X转换为BI-X


## 模型架构
args.mode_char='cnn' #'cnn'# 'cnn' #'cnn'# (None, lstm, cnn) # None: 不使用mode char
args.mode_word='lstm' #(lstm, cnn)
args.enable_crf=False


## 模型参数
args.dropout=0.5
dim_edb_char = 30 # char 嵌入维度
dim_emb_word = 100 # word 嵌入维度
dim_out_char = 10 #30 # 0 for No char repre # char representation 维度
dim_out_word = 512 # word representation 维度
args.dims=(dim_edb_char, dim_emb_word, dim_out_char, dim_out_word)

args.window_kernel=3 # cnn 参数


# 学习率
args.lr=1e-3
args.lr_step=10 # number of epoch for each lr downgrade
args.lr_gamma=0.1# strength of lr downgrade
args.eps_f1=1e-4 # minimum f1 score difference threshold



#%%
print('\n Load dataset and resources...')

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# get data
with open(args.path_processed, 'rb') as f: # 不过滤
    mappings = pickle.load(f)
# with open(args.path_filtered, 'rb') as f: # 过滤
#     mappings = pickle.load(f)

word2idx = mappings['word2idx']
char2idx = mappings['char2idx']
tag2idx = mappings['tag2idx']
glove_word = mappings['embeds_word']
glove_word = np.append(glove_word, [[0]*dim_emb_word], axis=0)


args.max_len_word = max([max([len(s) for s in word2idx.keys()]),37])  
# 37 is the longest length in testing set
args.idx_pad_char = max(char2idx.values()) + 1
args.idx_pad_word = max(word2idx.values()) + 1
args.idx_pad_tag = max(tag2idx.values()) + 1

print(f'\n[info] max seq len{args.max_len_word} | chars: {args.idx_pad_char} | words: {args.idx_pad_word} | tag: {args.idx_pad_tag}')
print(f'\n[info] shape of glove: {len(glove_word)}, {dim_emb_word}')


# get loader
from utils import load_sentences, update_tag_scheme, prepare_dataset
from torch.utils.data import DataLoader
from dataloader import CoNLLData, collate_fn

def get_data_loader(args, filename, zeros=digits_zero, is_lowercase=True, shuffle=True, batch_size=batch_size):
    _data = load_sentences(args.path_data+filename, zeros=zeros)
    update_tag_scheme(_data, args.tag_scheme)
    _data = prepare_dataset(_data, word2idx, char2idx, tag2idx, is_lowercase)

    if not zeros:
        return _data
    else:
        _loader = DataLoader(CoNLLData(args, _data), batch_size=batch_size, num_workers=num_worker, shuffle=shuffle, collate_fn=lambda x: collate_fn(x, args))
        return _loader

train_loader=get_data_loader(args, 'eng.train', zeros=digits_zero, is_lowercase=True)
valid_loader=get_data_loader(args, 'eng.testb', zeros=digits_zero, is_lowercase=True)
#%%


## get model
model = BiLSTM_CRF(args, word2idx, char2idx, tag2idx, glove_word).to(args.device)

args.criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)



# Print training parameters
n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
#%%
print(f'\n[info] | lr: {args.lr} | dropout: {args.dropout} |char: {args.mode_char} | word: {args.mode_word} | CRF: {args.enable_crf} '
        f'| Param: {n_param} | ')
print(model)

# begin training 
best_val_loss = 1e5
best_f1 = 0
best_epoch = 0
es_patience = 0

metric_his=[] # (trainf1, validf1)

count_time = []
#%%
for epoch in range(1,epochs+1):
    print('\n[Epoch {epoch}]'.format(epoch=epoch))

    t_start = time.time()
    loss_train, (f1_train, prec_train, rec_train) = train(args, model, train_loader, optimizer)
    scheduler.step()
    count_time.append(time.time() - t_start)
    print('  | Train | loss {:5.4f} | F1 {:5.4f} | {:5.2f} s |'.format(loss_train, f1_train,count_time[-1]))
    val_loss, (val_f1, val_prec, val_rec) = evaluate(args, model, valid_loader)

    # Save the model if the validation loss is the best we've seen so far.
    if val_f1 > best_f1:
        if val_f1 - best_f1 > args.eps_f1:
            es_patience = 0  # reset if beyond threshold
        with open(args.path_model, 'wb') as f:
            torch.save(model, f)
        best_val_loss = val_loss
        best_epoch = epoch
        best_f1 = val_f1
    else:
        # Early stopping condition
        es_patience += 1
        if es_patience >= es_patience_max:
            print('\n[Warning] Early stopping model')
            print('  | Best | Epoch {:d} | Loss {:5.4f} | F1 {:5.4f} |'
                    .format(best_epoch, best_val_loss, best_f1))
            break
    # logging
    print('  | Valid | loss {:5.4f} | F1 {:5.4f} | es_patience {:.0f} |'.format(val_loss, val_f1, es_patience))

    metric_his.append([(f1_train, prec_train, rec_train) ,
                   (val_f1,val_prec, val_rec)])

print(f"train time: {sum(count_time)}s; avg epoch: {np.average(count_time)}s")
#%%
## testing

import torch
from epoch import  evaluate

test_loader = get_data_loader(args, 'eng.testb')
# Load the best saved model and test
print('\n[Testing]')
with open(args.path_model, 'rb') as f:
    model = torch.load(f)
loss_test, (f1_test, prec_test, rec_test) = evaluate(args, model, test_loader)

print('  | Test | loss {:5.4f} | F1 {:5.4f} | Precision: {:5.4f} | recall: {:5.4f} '.format(loss_test, f1_test, prec_test, rec_test))
#%%
train_plot_dict = { "f1": [],
             "prec":[],
             "rec": []
             }
val_plot_dict = { "f1": [],
             "prec": [], 
             "rec": []
             }

for train_met, val_met in metric_his:
    
    _f1, _prec, _rec = train_met
    train_plot_dict['f1'].append(_f1)
    train_plot_dict['prec'].append(_prec)
    train_plot_dict['rec'].append(_rec)
    
    _f1, _prec, _rec = val_met
    val_plot_dict['f1'].append(_f1)
    val_plot_dict['prec'].append(_prec)
    val_plot_dict['rec'].append(_rec)
print(train_plot_dict, val_plot_dict)

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

'''
sns.set_theme(style="darkgrid")
df = sns.load_dataset("penguins")
sns.displot(
    df, x="flipper_length_mm", col="species", row="sex",
    binwidth=3, height=3, facet_kws=dict(margin_titles=True),
)
'''
def plot_(metric1:dict, metric2:dict, save=None):
    num_fig = 2
    sns.set_theme(style="darkgrid")
    plt.figure(num_fig, figsize=(5, 7))
    # plt.grid(True)
    
    plt.subplot(num_fig,1, 1)
    data_f = pd.DataFrame(metric1)
    sns.scatterplot(data=data_f)

    # plt.subplot(2)
    plt.subplot(num_fig,1, 2)
    data_f = pd.DataFrame(metric2)
    sns.lineplot(data=data_f)
    
    plt.savefig(save)

metric_f1 = {'train f1': train_plot_dict['f1'],
             'valid f1': val_plot_dict['f1']
             }
metric_prec = {
    'train precision': train_plot_dict['prec'],
    'valid precision': val_plot_dict['prec'],
    'train recall': train_plot_dict['rec'],
    'valid recall': val_plot_dict['rec']}

plot_(metric_f1, metric_prec, save=f'model/{args.mode_char}-{args.mode_word}-{args.enable_crf}.png')


#%%

def idx2word(i, word2idx:dict):
    for key,value in word2idx.items():
        if value == i:
            return key

def idx2tag(i, tag2idx:dict):
    for key,value in tag2idx.items():
        if value == i:
            return key

# write
from tqdm import tqdm
from sklearn.metrics import f1_score

def test_and_write(_data_iter,_data, type="test", model=model):
    preds= []

    pred_all =[]
    gt_all = []
        
    for _batch in tqdm(_data_iter, desc=f'  - {type}ing', leave=False):
        model.eval()
        words_batch, chars_batch, tags_batch, lens_batch = map(lambda x: x.to(args.device), _batch)
        
        with torch.no_grad():
            loss_batch, pred_batch = model.get_loss(words_batch, chars_batch, tags_batch, lens_batch)

            for pred, tags, _len in zip(pred_batch, tags_batch, lens_batch):
                preds.append(pred[:_len])
                
                gt_all += tags[:_len].cpu().tolist()
                pred_all += pred[:_len]

                
    f1 = f1_score(gt_all, pred_all, average='macro')
    print(f'# f1 score:', f1)


    if not write:
        return
    with open(f'model/{type}_res_{args.mode_char}_{args.mode_word}_{f1}_{args.enable_crf}_nodoc.txt',encoding="utf-8", mode='w') as f:
        idx=0
        for word_seq  in _data:
            word_seq = word_seq['str_words']
            
            if 'DOCSTART' in word_seq[0]:
                pred_seq = [tag2idx['O']]
                f.write('\n')
                
                continue
            else:
                pred_seq = preds[idx]
                idx+=1
            
            for w, t in zip(word_seq, pred_seq):
                f.write(w)
                f.write(' ')
                f.write(idx2tag(t, tag2idx))            
                f.write('\n')
            f.write('\n')
            
#%%
write = True
test_loader=get_data_loader(args, 'eng.testb', zeros=digits_zero, is_lowercase=True, shuffle=False, batch_size=1)
raw_test_data = get_data_loader(args, 'eng.testb', zeros=False, is_lowercase=False)
## 写出test结果到txt
test_and_write(test_loader, raw_test_data, type="test")

#%%
write = True

## evaluate
eval_loader=get_data_loader(args, 'eng.testa', zeros=digits_zero, is_lowercase=True, shuffle=False, batch_size=1)
raw_eval_data = get_data_loader(args, 'eng.testa', zeros=False, is_lowercase=False)

## 写出test结果到txt
test_and_write(eval_loader, raw_eval_data, type="evaluate")


#%%

args.enable_crf=True
model_crf = BiLSTM_CRF(args, word2idx, char2idx, tag2idx, glove_word).to(args.device)

args.criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, model_crf.parameters()), lr=args.lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

with open(args.path_model, 'rb') as f:
    model = torch.load(f)

print(model.state_dict().keys())
print(model_crf.state_dict().keys())


# # %%
# print(model_crf.state_dict()['transitions'])
# print(model_crf.state_dict()['conv1.bias'])

# for layer_ in model.state_dict().keys():
#     model_crf.state_dict()[layer_].copy_(model.state_dict()[layer_].clone().detach()) 

# print(model_crf.state_dict()['transitions'])
# print(model_crf.state_dict()['conv1.bias'])


# #%%
# # 微调model crf
# epochs_refine = 3

# best_val_loss = 1e5
# best_f1 = 0
# best_epoch = 0
# es_patience = 0

# f1_his_crf = []
# for epoch in range(1,epochs_refine+1):
#     print('\n[Epoch {epoch}]'.format(epoch=epoch))

#     t_start = time.time()
#     loss_train, f1_train = train(args, model_crf, train_loader, optimizer)
#     scheduler.step()
#     print('  | Train | loss {:5.4f} | F1 {:5.4f} | {:5.2f} s |'.format(loss_train, f1_train, time.time() - t_start))
#     val_loss, val_f1 = evaluate(args, model_crf, valid_loader)

#     # Save the model if the validation loss is the best we've seen so far.
#     if val_f1 > best_f1:
#         if val_f1 - best_f1 > args.eps_f1:
#             es_patience = 0  # reset if beyond threshold
#         with open("./result/models/model_refine.pt", 'wb') as f:
#             torch.save(model_crf, f)
#         best_val_loss = val_loss
#         best_epoch = epoch
#         best_f1 = val_f1
#     else:
#         # Early stopping condition
#         es_patience += 1
#         if es_patience >= es_patience_max:
#             print('\n[Warning] Early stopping model')
#             print('  | Best | Epoch {:d} | Loss {:5.4f} | F1 {:5.4f} |'
#                     .format(best_epoch, best_val_loss, best_f1))
#             break
#     # logging
#     print('  | Valid | loss {:5.4f} | F1 {:5.4f} | es_patience {:.0f} |'.format(val_loss, val_f1, es_patience))

#     f1_his_crf.append((f1_train ,val_f1))


# # %%
# write = True

# with open("./result/models/model_refine.pt", 'rb') as f:
#     model_crf = torch.load(f)

    
# test_loader=get_data_loader(args, 'eng.testb', zeros=digits_zero, is_lowercase=True, shuffle=False, batch_size=1)
# raw_test_data = get_data_loader(args, 'eng.testb', zeros=False, is_lowercase=False)
# ## 写出test结果到txt
# test_and_write(test_loader, raw_test_data, type="test", model=model_crf)