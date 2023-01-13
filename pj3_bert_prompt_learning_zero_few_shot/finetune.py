#%%
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2" # cuda:0, GPU1
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" # cuda:0, GPU1
from time import time

import dataset
import engine
import torch
import pandas as pd
import torch.nn as nn
import numpy as np
from model import BERTBaseUncased
from sklearn import model_selection
from sklearn import metrics

import transformers
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

#%%
class Config():
    def __init__(self) -> None:
        self.DEVICE = "cuda:0"
        self.MAX_LEN = 268
        self.TRAIN_BATCH_SIZE = 8
        self.VALID_BATCH_SIZE = 4
        self.EPOCHS = 10


        # 训练参数
        self.eps_thres=1e-4 
        self.es_max=5  # early stop

        self.BERT_PATH = "bert-base-uncased"

        self.MODEL_PATH = "/home/18307110500/pj3_workplace/pytorch_model.bin"

        self.TRAINING_FILE = "/home/18307110500/data/train_128.data"
        
        self.VALIDATION_FILE = "/home/18307110500/data/valid.data"
        self.TEST_FILE = "/home/18307110500/data/test.data"

        self.TOKENIZER = transformers.BertTokenizer.from_pretrained(self.BERT_PATH, do_lower_case=True)
        
config = Config()

#%%
# train_set = dataset.DatasetLoader(config.TRAINING_FILE)
# valid_set = dataset.DatasetLoader(config.VALIDATION_FILE)
_, train_dir= dataset.read_data(config.TRAINING_FILE)
_, valid_dir= dataset.read_data(config.VALIDATION_FILE)

train_dataset = dataset.BERTDataset(train_dir['x'], train_dir['y'],config=config)
valid_dataset = dataset.BERTDataset(valid_dir['x'], valid_dir['y'],config=config)
valid_data_loader = valid_dataset.get_dataloader(batch_size=config.VALID_BATCH_SIZE)
train_data_loader = train_dataset.get_dataloader(batch_size=config.TRAIN_BATCH_SIZE)


device = torch.device(config.DEVICE)
model = BERTBaseUncased(config)
model.to(device)
print(model)
param_optimizer = list(model.named_parameters())

no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
optimizer_parameters = [
    {
        "params": [
            p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
        ],
        "weight_decay": 0.001,
    },
    {
        "params": [
            p for n, p in param_optimizer if any(nd in n for nd in no_decay)
        ],
        "weight_decay": 0.0,
    },
]

num_train_steps = int(len(train_dir['x']) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
optimizer = AdamW(optimizer_parameters, lr=3e-5)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
)
#%%

# run
best_accuracy = 0
best_epoch = -1
early_stop_count = 0

ev_acc_his = []
tr_loss_his = []
tr_time_his=[]

#%%
for epoch in range(config.EPOCHS):
    tr_start = time()
    loss_tr_his =engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
    tr_loss_his.append(np.average(loss_tr_his))
    tr_time_his.append(time()-tr_start)
    
    outputs, targets, loss_ev_his = engine.eval_fn(valid_data_loader, model, device)
    outputs = np.array(outputs) >= 0.5
    accuracy = metrics.accuracy_score(targets, outputs)
    print(f"epoch: {epoch} | Train Loss: {np.average(loss_tr_his)} | Eval Loss: {np.average(loss_ev_his)} | Acc: {accuracy}")

    ev_acc_his.append(accuracy)
    
    if accuracy > best_accuracy:
        if accuracy - best_accuracy > config.eps_thres:
           early_stop_count = 0 # reset
        torch.save(model, f"fewshot32_seq{config.MAX_LEN}-hidden768-best.pth")
        best_accuracy = accuracy
        best_epoch = epoch
        best_time_eval_loss = np.average(loss_ev_his)
        
    else: 
        early_stop_count+=1
        if early_stop_count >= config.es_max:
            print('\n[Warning] Early stopping model')
            print('  | Best | Epoch {:d} | Acc {:5.4f} |'
                    .format(best_epoch,  best_accuracy))
            break
# %%
# eval
model = torch.load(f"fewshot32_seq{config.MAX_LEN}-hidden768-best.pth")
accuracy = best_accuracy
print(f"Accuracy Score = {accuracy}")
torch.save(model, f"fewshot32_seq{config.MAX_LEN}-hidden768-{accuracy}.pth")
print(f"saved as fewshot32_seq{config.MAX_LEN}-hidden768-{accuracy}.pth")

# %%
metric_rec = {
    'epo':[i+1 for i in range(len(ev_acc_his))],
    'eval acc': ev_acc_his,
    'train loss': tr_loss_his ,
    'epoch time(s)': tr_time_his
}
data_f = pd.DataFrame(metric_rec)
data_f
# %%
eval_acc = best_accuracy
avg_epo_time= np.average(tr_time_his)
print("fewshot {} | best acc {} | epoch {} | {:.3f}s".format(32, eval_acc, config.EPOCHS,avg_epo_time))
