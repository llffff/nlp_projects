#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7" 

from time import time
import random

import dataset
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import transformers
from  torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

#%%

class PromptConfig():
    def __init__(self, few_shot, BERT_PATH="bert-base-uncased", ) -> None:
        """
        few_shot=[None, '0', '32', '64', '128']
        # None for full data
        """
        self.few_shot = few_shot
        self.DEVICE = "cuda:1"
        self.MAX_LEN = 256
        self.TRAIN_BATCH_SIZE = 16
        self.VALID_BATCH_SIZE = 4
        self.train_times = 1
        self.EPOCHS = 2*self.train_times
        
        self.EARLY_STOP = 3
        self.eval_zero_shot = False # 是否测试zero shot
        self.test_output = False # 是否测试输出
        self.UPDATE_VERBAL = False # 是否更新verbalizer
        self.use_demostration = False # 是否使用 prompt type4 demostration
        self.random_demostration = False # 是否使用随机/人工选取的样本
        
        # 训练参数
        self.eps_thres=1e-4 
        self.es_max=5  # early stop

        self.BERT_PATH = BERT_PATH
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
        self.template =  " It is a {} film ."
        # self.template =  "It was {} ." # .format('[MASK]')
        # self.verbalizer=['terrible', 'great']
        self.candidate_ids = [self.TOKENIZER._convert_token_to_id(_) for _ in self.verbalizer]

        self.mask_id = self.TOKENIZER._convert_token_to_id(self.mask)
        
        
''' 基于 BertMaskedML的 few shot '''
paths= ["bert-base-uncased","bert-large-uncased", "albert-base-v2", "albert-large-v2", "roberta-base","roberta-large"]
# config = PromptConfig(BERT_PATH=paths[0], few_shot="32") # few shot
config = PromptConfig(BERT_PATH=paths[0], few_shot=None) # full data

#%%
# model: bert masked lm
model_bert = config.MODEL 
bert_tokenzier = config.TOKENIZER

bert_tokenzier: transformers.BertTokenizer

class PromptDataset(dataset.BERTDataset):
    def __init__(self, review, target, config):
        # self.config = config
        self.template = config.template # "It is a {} film." # [MASK]
        self.mask = config.mask# '[MASK]' # bert_tokenzier.mask_token
        self.sep = config.TOKENIZER.sep_token
        self.use_demostration = config.use_demostration
        super(PromptDataset, self).__init__(review, target, config)

    def _random_neg_pos(self, item):
        self_ = self
        item_i = item

        break_epo = 50
        rand_ids = [-1,-1] #0: negative
        
        
        while not (rand_ids[0]>-1 and rand_ids[1]>-1) and break_epo > 0:
            break_epo -= 1 
            
            label_ = 0 if rand_ids[0] == -1 else 1 # 判断是找pos的随机样本还是neg
            
            rand_i = random.randint(0, len(self_)-1) # 样本随机值
            if rand_i == item_i: # 不能等于自身
                continue
            if not self_.target[rand_i] == label_: # 需要指定pos或neg
                continue
            
            if break_epo <= 0:
                rand_ids[label_] = label_+1
                continue
            
            rand_ids[label_] = rand_i

        return rand_ids[0],rand_ids[1]

    
    def make_prompt(self, input_data, replace=config.mask):
        input_trans = " ".join([input_data, self.template.format(replace)])
        return input_trans

    # def _get_target(self,item,dtype=torch.float):
    #     # prompt的标签
    #     # logits # len(vocab)

    #     tar = super()._get_target(item, dtype=torch.long)
    #     tok_idx = self.config.candidate_ids[tar] # 0,1
    #     return torch.tensor(tok_idx, dtype=dtype)

    def getReview(self, item):
        neg_item, pos_item= 1,0
        if self.config.random_demostration:
            neg_item, pos_item = self._random_neg_pos(item)
        # demonstration
        
        # 样本
        review = super().getReview(item)
        review_trans = self.make_prompt(review)

        if not self.use_demostration:
            return review_trans
        
        # 随机负例
        review_neg = super().getReview(neg_item)
        review_trans_neg = self.make_prompt(review_neg, replace=self.config.verbalizer[0])
        
        # 随机正例
        review_pos = super().getReview(pos_item)
        review_trans_pos = self.make_prompt(review_pos,replace=self.config.verbalizer[1])

        # 随机正例先或负例先
        if random.randint(0,1) == 0:
            review_trans = f"{review_trans} {self.sep} {review_trans_neg} {self.sep} {review_trans_pos}"
        else:
            review_trans = f"{review_trans} {self.sep} {review_trans_pos} {self.sep} {review_trans_neg}"
        
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
def get_logits_of_mask(input_ids, logits, tok=config.mask, tokenzier=bert_tokenzier,ids=None):
    """
    Args:
        inputs_tok (tensor): 输入字符串经过tokenized得到的字典
        tok (str, optional): 可以是'[MASK]'或任意word token. Defaults to '[MASK]'.

    Returns:
        (tensor, tensor): 返回mask处的logits，返回mask的列索引

    Tips: 可以传入多个batch size

    Modify: 改为torch实现

    """
    if ids is None:
        # find_idx_of_tok_in_seq
        tok_id =  tokenzier._convert_token_to_id(tok)
        ids_of_mask_in_seq = torch.nonzero(input_ids == tok_id)[:,1] ## 得到mask的列索引
    else:
        ids_of_mask_in_seq = ids
        
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


#%%
def loss_fn_ce(outputs, targets):
    '''
    outputs = mask_logits
    target = [tok_neg, tok_pos]

    ce= nn.MultiLabelSoftMarginLoss()
    a1 = torch.tensor([[0.1,0.9,0.01], [0.2,0.8,0.001]])
    a2 = torch.tensor([1,2])
    l1=ce(a1.view(len(a1),-1),a2.view(-1,1))
    a2 = torch.tensor([2,2])
    l2=ce(a1.view(len(a1),-1),a2.view(-1,1))
    a2 = torch.tensor([1,1])
    l3=ce(a1.view(len(a1),-1),a2.view(-1,1))

    l1,l2,l3
    l1=-l1

    '''
    batch_size = len(targets)
    
    l = nn.MultiLabelSoftMarginLoss()(outputs.view(batch_size,-1), targets.view(-1, 1))
    
    return l


def get_logits(config, logits_mask):
    # init: candidate_ids = config.candidate_ids
    labels_pr = logits_mask[:, config.candidate_ids]
    return labels_pr


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
            logits_mask, mask_ids = get_logits_of_mask(d['ids'], logits, tok=config.mask ,tokenzier=bert_tokenzier,ids=d["mask_idx"])

            # optional: 计算整个vocab上的softmax score
            # logits_mask = torch.softmax(logits_mask, dim=-1)
            
            # 取出verbalizer对应的logits
            labels_pr = get_logits(config, logits_mask)
            pred = [np.argmax(_) for _ in labels_pr]

            # 更改targets
            # pred = [np.argmax(_) for _ in logits_mask]
                
            _targets.extend(d['targets'])
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

early_stop_count = 0
#%%
for epoch in range(config.EPOCHS//config.train_times):
        
    # begin training
    tr_time_s = time()

    tr_loss = []

    # config.train_times = 5
    for epo_tr in range(config.train_times):
        for bi, d in tqdm(enumerate(train_data_loader), total=len(train_data_loader)):
            
            model_bert.train()
            dict_keys = [("ids","input_ids" ),("token_type_ids","token_type_ids"),("mask","attention_mask")]
            
            input_token = {
                    k[1]: d[k[0]].to(device) 
                    for k in dict_keys}

            targets = d['targets']
            
            labels = d['ids'].clone()
            labels[labels!=config.mask_id] = -100
            
            for idx_ in range(len(labels)):
                mask_idx = d['mask_idx'][idx_]
                labels[idx_, mask_idx] = config.mask_id

            labels=labels.to(device)

            optimizer.zero_grad()
            res_batch = model_bert(**input_token, labels=labels, return_dict=True)

            loss = res_batch.loss
            logits = res_batch.logits.cpu()

            ''' 取出mask位置上，candidate label对应的logits '''
            # # mask 位置的预测logits
            # logits_mask, mask_ids = get_logits_of_mask(d['ids'], logits, tok=config.mask,tokenzier=bert_tokenzier, ids=d["mask_idx"])
            # # logits_mask: (batch_size, vocab_size)
            # ### 计算loss
            
            # # 取出verbalizer对应的logits
            # labels_pr = get_logits(config, logits_mask)
            # # 概率分数
            # labels_pr = torch.softmax(labels_pr, dim=-1)
            # # labels_pr: (batch_size, 2)
            # # 取出 positive 对应的分数 (negative = 1-positive)
            # pred = labels_pr[:,1]
            # loss = loss_fn(pred, targets)

            # 更改过targets的实现
            # loss = loss_fn_ce(logits_mask, targets)
            
            tr_loss.append(loss.cpu().detach().item())
            # print(loss) # 0.6433

            ### 反传
            loss.backward()
            optimizer.step()
            scheduler.step()

            torch.cuda.empty_cache()

            if (bi+1) % 25 == 0:
                # begin eval
                fin_targets_eval,fin_outputs_eval,fin_logits_eval,fin_mask_ids_eval = eval_prompt(valid_data_loader, model_bert, device)

                acc = count_acc(fin_outputs_eval,fin_targets_eval)
                print(f"[Eval] Acc:{acc} | pred.sum: {np.sum(fin_outputs_eval)} | target.sum: {np.sum(fin_targets_eval)}")


    tr_time_his.append((time()-tr_time_s)/config.train_times)
    tr_time_s = time()
    tr_loss_his.append(np.mean(tr_loss))

    # begin eval
    fin_targets_eval,fin_outputs_eval,fin_logits_eval,fin_mask_ids_eval = eval_prompt(valid_data_loader, model_bert, device)
    
    ev_acc_his.append(count_acc(fin_outputs_eval,fin_targets_eval))

    loss_str = "{:.4f}".format(tr_loss_his[-1])
    print(f"[Train] Epoch: {epoch}/{config.EPOCHS} | Train Loss: {loss_str} | Train time: {tr_time_his[-1]}s")
    print(f"[Eval] Acc:{ev_acc_his[-1]} | pred.sum: {np.sum(fin_outputs_eval)} | target.sum: {np.sum(fin_targets_eval)}")

    best_acc = max(ev_acc_his[:-1]) if epoch > 1 else -10
    if ev_acc_his[-1] > 0.5 and ev_acc_his[-1] > best_acc: # > best acc
        # torch.save(model_bert, f"fewshot{config.few_shot}-{config.BERT_PATH}-best.pth")
        print("[Best epoch]")
        # reset
        early_stop_count= 0
    if early_stop_count > config.EARLY_STOP: 
        print(f"[WARNING] early stop at epoch {epoch}.")
        break


# %%
idx = 100
if idx >= 0:
    logits = fin_logits_eval[idx]
    pred = fin_outputs_eval[idx]

    target = valid_dataset._get_target(idx)
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
    'epoch time(s)': tr_time_his
}
data_f = pd.DataFrame(metric_rec)
data_f
# %%
avg_epo_time= np.average(tr_time_his)
print("model {} | fewshot {} | best acc {} | epoch {} | {:.3f}s".format(config.BERT_PATH ,config.few_shot,eval_acc, config.EPOCHS,avg_epo_time))
