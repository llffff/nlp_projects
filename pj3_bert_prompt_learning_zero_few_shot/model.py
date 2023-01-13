import transformers
import torch.nn as nn
import torch

class BERTBaseUncased(nn.Module):
    def __init__(self, config, freeze_bert=False):
        super(BERTBaseUncased, self).__init__()
        # self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)
        self.bert = transformers.BertForMaskedLM.from_pretrained(config.BERT_PATH)
        
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

                
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 1)
        # self.out = nn.Linear(768*4, 1)

    def forward(self, ids, mask, token_type_ids):
        # _,output = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        # bo = self.bert_drop(output)
        # output = self.out(bo)

        # input: (batch_size, 64)
        output = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids,output_hidden_states=True)

        hidden_states = torch.cat(tuple([output.hidden_states[i] for i in [-1]]), dim=-1) # [batch_size, 64, 768*4]
        
        # 取 max
        # output = hidden_states[:, 0, :] # (batch_size, 1, 768*4) # MLM模型没有训练CLS?
        output = hidden_states.max(dim=1,keepdim=False)[0] # (batch_size, 64)
        
        bo = self.bert_drop(output)
        output = self.out(bo)  # (batch_size, 1)

        # print("ids.size()", ids.size()) 
        # print("bo.size()", bo.size())
        # print("output.size()", output.size())
        
        return output

class ALBERTBase(nn.Module):
    def __init__(self, config, freeze_bert=False):
        super(ALBERTBase, self).__init__()
        
        
        self.bert = transformers.BertForMaskedLM.from_pretrained(config.BERT_PATH)
        
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

                
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 1)
        # self.out = nn.Linear(768*4, 1)

    def forward(self, ids, mask, token_type_ids):
        # _,output = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        # bo = self.bert_drop(output)
        # output = self.out(bo)


        # input: (batch_size, 64)
        output = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids,output_hidden_states=True)

        hidden_states = torch.cat(tuple([output.hidden_states[i] for i in [-1]]), dim=-1) # [batch_size, 64, 768*4]
        # 取 cls token
        output = hidden_states[:, 0, :] # (batch_size, 1, 768*4)
        
        # output = output.max(dim=1,keepdim=False)[0] # (batch_size, 64)
        
        bo = self.bert_drop(output)
        output = self.out(bo)  # (batch_size, 1)

        # print("ids.size()", ids.size()) 
        # print("bo.size()", bo.size())
        # print("output.size()", output.size())
        
        return output

    
if __name__ == '__main__':
        
    
    # test_BertForMaskedLM(config)
    # test_BERTBaseUncased(config)
            
    #%%
    from transformers import AlbertTokenizer, AlbertForMaskedLM, AlbertModel

    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

    #%%
    model_masked = AlbertForMaskedLM.from_pretrained("albert-base-v2")
    print(model_masked)
    text = "Hello world"
    targets = ["This is computer","This is software"]
    
    encoded_input = [tokenizer([text, _], 
                                     add_special_tokens=True, padding=True, return_tensors='pt')
                           for _ in targets]


    outputs = [model_masked(**_[0],labels=_[1],
                            output_hidden_states=True) 
               for _ in encoded_input]

    # %%
    print(encoded_input)
    # 将两个pair当成大小为2的batch了
    # 0 padding; 2=[CLS], 3=[SPE], attention_mask表示长度mask
    # {'input_ids': tensor([[    2, 10975,   126,     3,     0],
    #     [    2,    48,    25,  1428,     3]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 0],
    #     [1, 1, 1, 1, 1]])}
    
    output = outputs[0]
    print(output)
    print(output.logits.size())
    # torch.Size([2, 6, 30000])
    
    #%%
    model_norm = AlbertModel.from_pretrained("albert-base-v2")
    output = model_norm(**encoded_input) 
    print("model_norm.output:", output)
    print("last_hidden_state:", output.last_hidden_state.size())
    print("pooler_output:",output.pooler_output.size())
    # torch.Size([1, 5, 768])
    # torch.Size([1, 768])
    
    