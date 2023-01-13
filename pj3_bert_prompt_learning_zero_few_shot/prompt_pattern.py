#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7" 
from time import time

import dataset
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import transformers
from  torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from collections import Counter

#%%

class PromptConfig():
    def __init__(self, few_shot, BERT_PATH="bert-base-uncased", ) -> None:
        """
        few_shot=[None, '0', '32', '64', '128']
        # None for full data
        """
        self.few_shot = few_shot
        self.DEVICE = "cuda:5"
        self.MAX_LEN = 256
        self.TRAIN_BATCH_SIZE = 8
        self.VALID_BATCH_SIZE = 4
        self.train_times = 1
        self.EPOCHS = 1*self.train_times
        
        self.EARLY_STOP = 3
        
        self.eval_zero_shot = False # 是否测试zero shot
        self.test_output = False # 是否测试输出
        self.UPDATE_VERBAL = 0
        # =0 不更新verbalizer
        # >0 每多少epoch更新一次verbalizer
        
        
        # 训练参数
        self.eps_thres=1e-4 
        self.es_max=5  # early stop

        self.BERT_PATH = BERT_PATH
        self.MODEL_PATH = "/home/18307110500/pj3_workplace/pytorch_model.bin"
        data_dir ="/home/18307110500/data"

        if few_shot is not None and few_shot == '0':
            self.TRAINING_FILE = None
        elif few_shot is not None and os.path.exists(f"{data_dir}/train_{few_shot}.data"):
            self.TRAINING_FILE =f"{data_dir}/train_{few_shot}.data"
        else:
            self.TRAINING_FILE = f"{data_dir}/train.data"

        self.VALIDATION_FILE = f"{data_dir}/valid.data"
        self.TEST_FILE = f"{data_dir}/test.data"

        BERT_PATH:str
        if BERT_PATH.startswith("bert"):
            self.TOKENIZER= transformers.BertTokenizer.from_pretrained(self.BERT_PATH, do_lower_case=True)
            self.MODEL = transformers.BertForMaskedLM.from_pretrained(self.BERT_PATH)
            
        elif BERT_PATH.startswith("albert"):
            # AlbertTokenizer, AlbertForMaskedLM
            self.TOKENIZER= transformers.AlbertTokenizer.from_pretrained(self.BERT_PATH, do_lower_case=True)
            self.MODEL = transformers.AlbertForMaskedLM.from_pretrained(self.BERT_PATH)
            
        elif BERT_PATH.startswith("roberta"):
            # RobertaTokenizer, RobertaForMaskedLM
            self.TOKENIZER= transformers.RobertaTokenizer.from_pretrained(self.BERT_PATH, do_lower_case=True)
            self.MODEL = transformers.RobertaForMaskedLM.from_pretrained(self.BERT_PATH)
            
        # prompt
        # label转换为id
        self.mask = self.TOKENIZER.mask_token # '[MASK]'/'<mask>'
        # self.verbalizer=['negative', 'positive']
        self.verbalizer=['bad', 'great']
        self.template =  "It is a {} film ."
        # self.template =  "It was {} ." # .format('[MASK]')
        # self.verbalizer=['terrible', 'great']
        self.candidate_ids = [self.TOKENIZER._convert_token_to_id(_) for _ in self.verbalizer]
        
''' 基于 BertMaskedML的 few shot '''
paths= ["bert-base-uncased","bert-large-uncased", "albert-base-v2", "albert-large-v2", "roberta-base","roberta-large"]
config = PromptConfig(BERT_PATH=paths[0], few_shot=None) # few shot

#%%
# model: bert masked lm
model_bert = config.MODEL 
bert_tokenzier = config.TOKENIZER

bert_tokenzier: transformers.BertTokenizer

class PromptDataset(dataset.BERTDataset):
    def __init__(self, review, target, config):

        self.template = config.template # "It is a {} film." # [MASK]
        self.mask = config.mask# '[MASK]' # bert_tokenzier.mask_token
        super(PromptDataset, self).__init__(review, target, config)
        
    # sep = bert_tokenzier.sep_token
    def make_prompt(self, input_data):
        input_trans = f"{input_data} {self.template.format(self.mask)}"
        return input_trans

    def getReview(self, item):
        review = super().getReview(item)
        review_trans = self.make_prompt(review)
        return review_trans
        

_, train_dir= dataset.read_data(config.TRAINING_FILE)
_, valid_dir= dataset.read_data(config.VALIDATION_FILE)

train_dataset = PromptDataset(train_dir['x'], train_dir['y'],config=config)
valid_dataset = PromptDataset(valid_dir['x'], valid_dir['y'],config=config)

valid_data_loader = valid_dataset.get_dataloader(batch_size=config.VALID_BATCH_SIZE)
train_data_loader = train_dataset.get_dataloader(batch_size=config.TRAIN_BATCH_SIZE)

print(train_dataset.getReview(0), train_dataset.target[0])
print(valid_dataset.getReview(0), valid_dataset.target[0])
# "nothing about the film -- with the possible exception of elizabeth hurley 's breasts -- is authentic .  It is a [MASK] review." 0

#%%
def get_logits_of_mask(input_ids, logits, tok=config.mask, tokenzier=bert_tokenzier):
    """
    Args:
        inputs_tok (tensor): 输入字符串经过tokenized得到的字典
        tok (str, optional): 可以是'[MASK]'或任意word token. Defaults to '[MASK]'.

    Returns:
        (tensor, tensor): 返回mask处的logits，返回mask的列索引

    Tips: 可以传入多个batch size

    Modify: 改为torch实现

    """
    # find_idx_of_tok_in_seq
    tok_id =  tokenzier._convert_token_to_id(tok)
    ids_of_mask_in_seq = torch.nonzero(input_ids == tok_id)[:,1] ## 得到mask的列索引
    
    # convert to tensor
    logits_tok = torch.stack([logits[idx, ids_of_mask_in_seq[idx],:]for idx in range(logits.size(0))])

    # logits_tok.size() # [4, 30522]=[batch size, vocab size]
    return logits_tok, ids_of_mask_in_seq


''' train: fine tune bert '''

def count_acc(pred, target):
    acc_count = np.sum(np.array(pred) == np.array(target))
    return acc_count/len(pred)

def loss_fn(outputs, targets):
    # sigmoid + cross entropy
    # print(outputs, targets)
    return nn.BCEWithLogitsLoss()(outputs.view(-1,1), targets.view(-1, 1))


def get_logits(config, logits_mask):
    # init: candidate_ids = config.candidate_ids
    labels_pr = logits_mask[:, config.candidate_ids]
    return labels_pr

def get_topk_token_ids(logits_mask,top_k =10):
    logits_mask_=logits_mask.detach()
    batch_size = logits_mask_.size(0)
    idsk = []
    logitsk = []
    
    for i in range(batch_size):
        
        top_inds = list(reversed(np.argsort(logits_mask_[i].numpy(), axis=-1)))  # list
        idsk.append(top_inds[:top_k])
        logitsk.append(logits_mask_[i,top_inds][:top_k].numpy().tolist())
        
    return idsk, logitsk # (list, list) size=(bz, k)

        
def show_topk_cloze(logits_mask,top_k =10):
    # 根据logits排序
    top_inds = list(reversed(np.argsort(logits_mask)))
    res_top_k = []
    for i in top_inds:
        res_i = {
            "token_id":i.item(),
            "token_str": bert_tokenzier._convert_id_to_token(i.item()),
            "raw_score": logits_mask[i.item()] # 未经过softmax的分数
            }
        res_top_k.append(res_i)
        if len(res_top_k) >= top_k:
            break

    return res_top_k # 查看top k预测的填空

#%%

def eval_prompt(data_loader, model, device):
    _targets = []
    _outputs = []
    _logits = []
    _mask_ids = []
    model.eval()
    
    with torch.no_grad():
        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            dict_keys = [("ids","input_ids" ),("token_type_ids","token_type_ids"),("mask","attention_mask")]
            
            input_token = {
                k[1]: d[k[0]].to(device) 
                for k in dict_keys}

            res_batch = model(**input_token)
            
            logits = res_batch.logits
            logits = logits.cpu()
            logits_mask, mask_ids = get_logits_of_mask(d['ids'], logits, tok=config.mask ,tokenzier=bert_tokenzier)

            # optional: 计算整个vocab上的softmax score
            # logits_mask = torch.softmax(logits_mask, dim=-1)
            
            # 取出verbalizer对应的logits
            labels_pr = get_logits(config, logits_mask)
            pred = [np.argmax(_) for _ in labels_pr]
                
            _targets.extend(d['targets'].numpy().tolist())
            _outputs.extend(pred)
            _logits.extend(logits_mask)
            _mask_ids.extend(mask_ids)
            torch.cuda.empty_cache()


                
            # break
    return _targets,_outputs,_logits,_mask_ids

#%%
device = config.DEVICE
model_bert.to(device)

# zero
if config.eval_zero_shot:
    fin_targets_eval,fin_outputs_eval,fin_logits_eval,fin_mask_ids_eval = eval_prompt(valid_data_loader, model_bert, device)
    
    print("[Zero shot]")
    print(f"[Eval] Acc:{count_acc(fin_outputs_eval,fin_targets_eval)} | pred.sum: {np.sum(fin_outputs_eval)} | target.sum: {np.sum(fin_targets_eval)}")

#%%
param_optimizer = list(model_bert.named_parameters())
no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
# for _,__ in param_optimizer:
#     print(_)
optimizer_parameters = [ {
        "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) ],
        "weight_decay": 0.001,
    },
        {"params": [ p for n, p in param_optimizer if any(nd in n for nd in no_decay) ],
        "weight_decay": 0.0,
    },
]

print("opt param: ",len(optimizer_parameters[0]['params']))
print("no opt",len(optimizer_parameters[1]['params']))

#%%
num_train_steps = int(len(train_dir['x']) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
if len(optimizer_parameters[0]['params']) > 20:
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
else:
    # albert
    optimizer = AdamW(model_bert.parameters(), lr=3e-5)
    
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
)

#%%
ev_acc_his = []
tr_loss_his = []
tr_time_his=[]

tr_verb_his=[]
early_stop_count = 0

#test
# config.EPOCHS=config.train_times=1
#test
        
for epoch in range(config.EPOCHS//config.train_times):
    # begin training
    model_bert.train()
    tr_time_s = time()

    tr_loss = []

    # config.train_times = 5
    for epo_tr in range(config.train_times):
        for bi, d in tqdm(enumerate(train_data_loader), total=len(train_data_loader)):
            dict_keys = [("ids","input_ids" ),("token_type_ids","token_type_ids"),("mask","attention_mask")]
            targets = d['targets']
            input_token = {
                    k[1]: d[k[0]].to(device) 
                    for k in dict_keys}

            optimizer.zero_grad()
            res_batch = model_bert(**input_token)
            logits = res_batch.logits.cpu()

            ''' 取出mask位置上，candidate label对应的logits '''
            # mask 位置的预测logits
            logits_mask, mask_ids = get_logits_of_mask(d['ids'], logits, tok=config.mask,tokenzier=bert_tokenzier)
            # logits_mask: (batch_size, vocab_size)

            ### 计算loss
            
            # 取出verbalizer对应的logits
            labels_pr = get_logits(config, logits_mask)
            # 概率分数
            labels_pr = torch.softmax(labels_pr, dim=-1)
            # labels_pr: (batch_size, 2)
            # 取出 positive 对应的分数 (negative = 1-positive)
            pred = labels_pr[:,1]
            loss = loss_fn(pred, targets)
            
            tr_loss.append(loss.cpu().detach().item())
            # print(loss) # 0.6433

            ### 反传

            loss.backward()
            optimizer.step()
            scheduler.step()

            torch.cuda.empty_cache()
            if (bi+1) % 8 == 0:
                 # begin eval
                fin_targets_eval,fin_outputs_eval,fin_logits_eval,fin_mask_ids_eval = eval_prompt(valid_data_loader, model_bert, device)
                
                acc =count_acc(fin_outputs_eval,fin_targets_eval)

                print(f"[Eval] Acc:{acc} | pred.sum: {np.sum(fin_outputs_eval)} | target.sum: {np.sum(fin_targets_eval)}")

                

    tr_time_his.append((time()-tr_time_s)/config.train_times)
    tr_loss_his.append(np.mean(tr_loss))

    # begin eval
    fin_targets_eval,fin_outputs_eval,fin_logits_eval,fin_mask_ids_eval = eval_prompt(valid_data_loader, model_bert, device)
    
    ev_acc_his.append(count_acc(fin_outputs_eval,fin_targets_eval))

    loss_str = "{:.4f}".format(tr_loss_his[-1])
    print(f"[Train] Epoch: {epoch}/{config.EPOCHS} | Train Loss: {loss_str} | Train time: {tr_time_his[-1]}s")
    print(f"[Eval] Acc:{ev_acc_his[-1]} | pred.sum: {np.sum(fin_outputs_eval)} | target.sum: {np.sum(fin_targets_eval)}")

    ### 更新verbalizer
    # 仅计算loss，选取令loss最小的verbal，不更新梯度
    model_bert.eval()
    tars,outputs,mask_logits,mask_ids = eval_prompt(train_data_loader, model_bert, device)
    tars = torch.tensor(tars)

    # mask_logits: (train size, vocab size)
    
    # 从mask logits中计算top k
    mask_logits = torch.stack(mask_logits)
    topk_ids_all, topk_score_all = get_topk_token_ids(mask_logits)
        
    # (train size, k)
    
    # 计算loss
    def get_new_verbels(cnt_pos:Counter, cnt_neg:Counter, find_n=5, tokenizer=config.TOKENIZER):
        """
        返回list of {token index, token str}
        """
        verbels_pos =cnt_pos.most_common(find_n)
        verbels_pos=[{
            'tok_id':_[0], 
            'tok_str': tokenizer._convert_id_to_token(_[0])}
            for _ in verbels_pos ]

        verbels_neg =cnt_neg.most_common(find_n)
        verbels_neg=[{
            'tok_id':_[0], 
            'tok_str': tokenizer._convert_id_to_token(_[0])}
            for _ in verbels_neg]
        
        return verbels_pos, verbels_neg

    topk_ids_all = torch.tensor(topk_ids_all)
    topk_score_all = torch.tensor(topk_score_all)
    
    pos_idx = tars == 1
    pos_idx = torch.nonzero(pos_idx)
    topk_ids_pos = topk_ids_all[pos_idx]
    topk_score_pos = topk_score_all[pos_idx]
    
    neg_idx = tars == 0
    neg_idx = torch.nonzero(neg_idx)
    topk_ids_neg = topk_ids_all[neg_idx]
    topk_score_neg = topk_score_all[neg_idx]
    
    cnt_pos = np.array(topk_ids_pos).reshape(-1).tolist()
    cnt_pos = Counter(cnt_pos)
    cnt_pos.most_common(3)
    
    cnt_neg = np.array(topk_ids_neg).reshape(-1).tolist()
    cnt_neg = Counter(cnt_neg)


    ### 1: most freq
    pos_v, neg_v = get_new_verbels(cnt_pos,cnt_neg)
    print(pos_v, neg_v)
    
    ### 计算损失
    def get_loss(config, logits_mask, targets):
        """
        logits_mask: (batch_size, )
        """
        # 取出verbalizer对应的logits
        labels_pr = get_logits(config, logits_mask)
        # 概率分数
        labels_pr = torch.softmax(labels_pr, dim=-1)
        # 取出 positive 对应的分数 (negative = 1-positive)
        pred = labels_pr[:,1]
        loss = loss_fn(pred, targets)
        return loss
    

    def get_new_verbelizer(config, verbelizers, mask_logits, tars, pos_or_neg="pos"):
        temp = config.candidate_ids

        # compute loss
        losses = []
        for verbelizer_ in verbelizers:
            if pos_or_neg == 'pos':
                config.candidate_ids = [temp[0], verbelizer_['tok_id']]
            else:
                config.candidate_ids = [verbelizer_['tok_id'],temp[1]]
            l_ = get_loss(config, mask_logits,targets=tars)
            losses.append(l_.detach().item())
        config.candidate_ids = temp

        # post process
        losses = torch.tensor(losses)

        i = torch.argmin(losses)
        verbelizers[i]['loss'] = losses[i]
        
        print(pos_or_neg, verbelizers[i])
        tok_id_min_loss = verbelizers[i]['tok_id']

        return tok_id_min_loss
    
        
    ### 2: argmin loss Z[y]
    new_pos_tok_id = get_new_verbelizer(config, verbelizers=pos_v, mask_logits=mask_logits,tars=tars, pos_or_neg="pos")
    
    new_neg_tok_id = get_new_verbelizer(config, verbelizers=neg_v, mask_logits=mask_logits,tars=tars, pos_or_neg="neg")
        



    best_acc = max(ev_acc_his[:-1]) if epoch > 1 else -10
    if ev_acc_his[-1] > best_acc: # > best acc
        torch.save(model_bert, f"fewshot{config.few_shot}-{config.BERT_PATH}-best.pth")
        print("[Best epoch]")
        # reset
        early_stop_count= 0
    else:
        early_stop_count+=1
        
    if early_stop_count >= config.EARLY_STOP: 
        print(f"[WARNING] early stop at epoch {epoch}.")
        break

    if config.UPDATE_VERBAL>0 and ((epoch+1) % config.UPDATE_VERBAL)==0:
        config.verbalizer = [bert_tokenzier._convert_id_to_token(new_neg_tok_id),bert_tokenzier._convert_id_to_token(new_pos_tok_id) ]
        config.candidate_ids = [new_neg_tok_id, new_pos_tok_id]
        print(f"[Update verbalizer] {config.verbalizer}")

    tr_verb_his.append(str(config.verbalizer))


# %%
idx = 100
if idx >= 0:
    logits = fin_logits_eval[idx]
    pred = fin_outputs_eval[idx]

    target = valid_dataset.target[idx]
    sequence = valid_dataset.getReview(idx)
    ids = valid_dataset[idx]['ids']
    print(f"sequence: \'{sequence}\', target: {target}, pred: {pred}", show_topk_cloze(logits, top_k=10))

#%%
# 1. 测试
eval_acc = max(ev_acc_his) # best eval

if config.test_output:
    model = torch.load(f"fewshot{config.few_shot}-{config.BERT_PATH}-best.pth")
    _, test_dir= dataset.read_data(config.TEST_FILE, test=True)
    test_dataset = PromptDataset(test_dir['x'], test_dir['y'],config=config)
    test_data_loader = test_dataset.get_dataloader(batch_size=config.VALID_BATCH_SIZE)

    test_record = eval_prompt(test_data_loader, model, device)
    # targets ,outputs ,logits ,mask_ids
    test_preds = test_record[1]

    # 2. open文件写入结果
    with open(f'saved/few_shot{config.few_shot}_eval{eval_acc}_res.txt',encoding="utf-8", mode='w') as f:
        for pred in test_preds:
            f.write("positive" if pred==1 else 'negative')
            f.write('\n')
    print("Testing finish. Test results saved.")
#%%
metric_rec = {
    'epo':[(i+1)*config.train_times for i in range(len(ev_acc_his))],
    'eval acc': ev_acc_his,
    'train loss': tr_loss_his ,
    'epoch time(s)': tr_time_his,
    'verbalizer': tr_verb_his
}
data_f = pd.DataFrame(metric_rec)
data_f
# %%
avg_epo_time= np.average(tr_time_his)
print("model {} | fewshot {} | best acc {} | epoch {} | {:.3f}s".format(config.BERT_PATH ,config.few_shot,eval_acc, config.EPOCHS,avg_epo_time))
