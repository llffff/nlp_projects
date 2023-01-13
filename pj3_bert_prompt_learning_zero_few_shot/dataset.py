#%%
import re
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


from torch.utils.data.dataloader import DataLoader
from torch.utils.data import RandomSampler, SequentialSampler

#%%

class BERTDataset:
    def __init__(self, review, target, config):
        self.review = review
        self.target = target
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN
        self.config  = config

    def __len__(self):
        return len(self.review)


    def getReview(self, item):
        # review = str(self.review[item])
        # review = " ".join(review[item].split())
        review = " ".join(self.review[item])
        return review

    def __getitem__(self, item):
        review = self.getReview(item)

        inputs = self.tokenizer.encode_plus(
            review,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            padding = 'max_length',
            truncation=True # 消除warning
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(self.target[item], dtype=torch.float),
        }
        
    def get_dataloader(self, batch_size):
        dataloader = torch.utils.data.DataLoader(self, batch_size=batch_size, num_workers=4)
        return dataloader
        
def read_data(path, test=False):
    data = []
    with open(path, 'r', encoding="utf-8") as f:
        for line in f.readlines():
            if test:
                raw_words = line.strip() # test只有数据
            else:
                raw_words, target = line.strip().split("\t")
                target = 1 if target == 'positive' else 0

            # simple preprosessing
            def precessing(s):
                s = re.sub("[.,!?;\\/'-]", " ", s)
                return s.lower()

            if raw_words.endswith("\""):
                raw_words = raw_words[:-1]
                
            if raw_words.startswith('"'):
                raw_words = raw_words[1:]
                
            raw_words = raw_words.replace('""', '"')

            # optional: 替换标点
            # raw_words = precessing(raw_words)
            
            raw_words = raw_words.split(" ")
            # raw_words = filter(lambda _: len(_)>0, raw_words)
            

            if test: 
                data.append({
                    'raw_words': raw_words, 
                    'target':0 # 测试数据无target
                    })

            else:
                data.append({
                    'raw_words': raw_words, 
                    'target': int(target)
                    })

    print("# samples: {}".format(len(data)))

    # 处理为dict的格式
    data_dic = {'x': [], 'y':[]}
    for data_ in data :
        data_dic['x'].append(data_['raw_words'])
        data_dic['y'].append(data_['target'])
        
    return data, data_dic
