import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data.dataloader import DataLoader
import re
import numpy as np

import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(42)


def custom_collate(batch):
    '''padding'''

    input_idx = [_['input_idx'] for _ in batch]
    target = [_['target'] for _ in batch]

    ZERO_PAD = 0
    max_len = 100 
    # max_len = max(len(x) for x in input_idx)
    # 截断或者补0


    def random_order_len(total_len, max_len):
        b = np.arange(0,total_len,1,dtype=np.int16)
        np.random.shuffle(b)
        b = b[:max_len]
        b.sort()
        return b

    def get_random(tokens, max_len):
        order = random_order_len(len(tokens), max_len)
        new_token = []
        for i in order:
            new_token.append(tokens[i])
        return new_token

    batch_input_idx = [
        get_random(_, max_len)
        if len(_) > max_len 
        else _ + ([ZERO_PAD] *  (max_len - len(_))) 
        for _ in input_idx]
    
    # 转换tensor
    return torch.LongTensor(batch_input_idx), torch.LongTensor(target)


def load_pretrained_embedding(words, pretrained_vocab):
    '''从训练好的vocab中提取出words对应的词向量'''
    embed = torch.zeros(len(words), pretrained_vocab.vectors[0].shape[0]) # torch.Size([100])
    oov_count = 0 # out of vocabulary
    save_oov = []
    for i, word in enumerate(words):
        try:
            idx = pretrained_vocab.stoi[word]
            embed[i, :] = pretrained_vocab.vectors[idx] # 将第i行用预训练的单词向量替换
        except KeyError:
            oov_count += 1
            save_oov.append(word)
            
    if oov_count > 0:
        print("There are %d oov words." % oov_count)
        print(save_oov[:100])
    return embed


def build_vocab(datasets):
    vocab ={'<pad>':0, '<unk>':1}
    idx_to_word = {0:'<pad>', 1:'<unk>'}
    for set in datasets:
        for ins in set:
            for word in ins['raw_words']:
                if word not in vocab:
                    idx_to_word[len(vocab)] = word
                    vocab[word] = len(vocab)

    return vocab, idx_to_word



from tqdm import tqdm
import os
import random

class ImdbDatasetLoader(Dataset):
    def __init__(self, path, test=False) -> None:
        super().__init__()

        self.data = []
        
        for label in ['pos', 'neg']:
            
            folder_name = os.path.join(path, label)
            target = 1 if label == 'pos' else 0
            
            for file in tqdm(os.listdir(folder_name)): 
                # os.listdir(folder_name) 读取文件路径下的所有文件名，并存入列表中
                
                with open(os.path.join(folder_name, file), 'rb') as f:
                    review = f.read().decode('utf-8').replace('\n', ' ')
                    tokens = [tok.lower() for tok in review.split(' ')]
                    
                    if test: 
                        self.data.append({
                            'raw_words': tokens, 
                            'target': 0
                            })

                    else:
                        self.data.append({
                            'raw_words': tokens, 
                            'target': int(target)
                            })

        random.shuffle(self.data) # 打乱data列表中的数据排列顺序


        self.data = self.data[:8000]

        print("# samples: {}".format(len(self.data)))


    def __len__(self):
        return len(self.data)


    def __getitem__(self, item):
        return self.data[item]

    def convert_word_to_idx(self, vocab):
        for i in range(len(self.data)):
            inx = self.data[i]
            word_idx = [
                vocab[x] 
                if x in vocab 
                else vocab['<unk>'] 
                for x in inx['raw_words']]

            self.data[i]['input_idx'] = word_idx

    

    def get_random_iter(self, batch_size, num_workers):        
        sampler = RandomSampler(data_source=self)
        return  DataLoader(self, 
                           batch_size=batch_size, sampler=sampler,
                           num_workers=num_workers, collate_fn=custom_collate)


    def get_sequential_iter(self, batch_size, num_workers ):
        sampler = SequentialSampler(data_source=self)
        return  DataLoader(self,
                           batch_size=batch_size, 
                           sampler=sampler, 
                           num_workers=num_workers, collate_fn=custom_collate)


# 创建词典
def get_vocab_imdb(datasets):
    d = []

    for set in datasets:
        for ins in set:
            d.extend([word for word in ins['raw_words'] ])
    counter = collections.Counter(d)
    
    
    vocab ={'<pad>':0, '<unk>':1}
    idx_to_word = {0:'<pad>', 1:'<unk>'}

    for word in counter:
        if counter.get(word)>5:
            vocab[word] = len(vocab)
            idx_to_word[len(vocab)] = word


    return vocab, idx_to_word




class DatasetLoader(Dataset):
    def __init__(self, path, test=False) -> None:
        super().__init__()

        self.data = []
        with open(path, 'r', encoding="utf-8") as f:
            for line in f.readlines():
                if test:
                    raw_words = line.strip()
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
                # 替换标点
                raw_words = precessing(raw_words)
                raw_words = raw_words.split(" ")
                # raw_words = filter(lambda _: len(_)>0, raw_words)
                

                if test: 
                    self.data.append({
                        'raw_words': raw_words, 
                        'target':0
                        })

                else:
                    self.data.append({
                        'raw_words': raw_words, 
                        'target': int(target)
                        })
        print("# samples: {}".format(len(self.data)))



    def __len__(self):
        return len(self.data)


    def __getitem__(self, item):
        return self.data[item]


    def convert_word_to_idx(self, vocab):
        for i in range(len(self.data)):
            inx = self.data[i]
            word_idx = [
                vocab[x] 
                if x in vocab 
                else vocab['<unk>'] 
                for x in inx['raw_words']]

            self.data[i]['input_idx'] = word_idx


    def get_random_iter(self, batch_size, num_workers):        
        sampler = RandomSampler(data_source=self)
        return  DataLoader(self, 
                           batch_size=batch_size, sampler=sampler,
                           num_workers=num_workers, collate_fn=custom_collate)


    def get_sequential_iter(self, batch_size, num_workers ):
        sampler = SequentialSampler(data_source=self)
        return  DataLoader(self,
                           batch_size=batch_size, 
                           sampler=sampler, 
                           num_workers=num_workers, collate_fn=custom_collate)



if __name__ == "__main__":
    data_dir = "../data/fudan_nlp/"
    base_dir = "/home/mlsnrs/data/data/lff/ai-lab/lab3/"

    train_set = DatasetLoader(f"{data_dir}/train.data")
    dev_set = DatasetLoader(f"{data_dir}/valid.data")


    print(" -- training set len: %d" % len(train_set))
    print(" -- training set data0.raw : %s" % train_set[0]['raw_words'])
    print(" -- training set data0.#words : %d" % len(train_set[0]['raw_words']))
    print(" -- training set data0.target: %s" % train_set[0]['target'])


    '''
    # samples: 8596
    # samples: 1000
    -- training set len: 8596
    -- training set data0.raw : ['light-years', 'ahead', 'of', 'paint-by-number', 'american', 'blockbusters', 'like', 'pearl', 'harbor', ',', 'at', 'least', 'artistically', '.']
    -- training set data0.#words : 14
    -- training set data0.target: 
    '''