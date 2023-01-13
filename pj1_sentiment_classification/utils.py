import torch
import dataset_utils as data
from sklearn.metrics import accuracy_score
from tqdm import tqdm


def eval(model, dev_set:data.DatasetLoader, batch_size, num_workers, device):
    model.eval()

    eval_loss = 0
    eval_pred, eval_y = [],[]
    
    dev_data_iter = dev_set.get_sequential_iter(batch_size, num_workers)
    
    for step, dev_batch in enumerate(dev_data_iter):
        x,y = dev_batch
        with torch.no_grad():
            loss, pred = model(x.to(device), y.to(device))
            pred = pred.detach().cpu().numpy().tolist()
                
            eval_loss+= loss.cpu().item()
            eval_pred.extend(pred)
            eval_y.extend(y)

    return eval_loss/step, accuracy_score(eval_pred, eval_y), eval_pred, eval_y



def train_batch(model, train_set:data.DatasetLoader,
                batch_size, num_workers, device,
                optimizer, weighted = 1.0):
    model.train()        
    
    tr_loss = 0
    tr_pred, tr_y = [],[]
    
    train_data_iter=train_set.get_random_iter(num_workers=num_workers, batch_size=batch_size)
    for step, batch in enumerate(tqdm(train_data_iter)):
        # forward
        x,y=batch
        # print(len(x),len(y))
        loss, pred = model(x.to(device), y.to(device))

        loss = weighted*loss
        
        # backward
        loss.backward()
        optimizer.step()
        model.zero_grad()

        tr_loss += loss.cpu().item()
        tr_pred.extend(pred.detach().cpu().numpy().tolist())
        tr_y.extend(y)

    return tr_loss/step, accuracy_score(tr_pred, tr_y)




def train_batch_cross_domain(
    model, train_set:data.DatasetLoader,
    train_set_imdb:data.ImdbDatasetLoader,
                batch_size, num_workers, device,
                optimizer, weighted = 1.0):
    model.train()        
    
    tr_loss = 0
    tr_pred, tr_y = [],[]
    
    aux_loss = 0
    aux_pred, aux_y = [],[]
    
    train_data_iter=train_set.get_random_iter(num_workers=num_workers, batch_size=batch_size)
    
    train_imdb_data_iter=train_set_imdb.get_random_iter(num_workers=num_workers, batch_size=batch_size*2)
    
    for step, batch in enumerate(tqdm(train_data_iter)):
        # forward
        x,y=batch
        loss, pred = model(x.to(device), y.to(device))
        # backward
        loss.backward()
        optimizer.step()
        model.zero_grad()

        tr_loss += loss.cpu().item()
        tr_pred.extend(pred.detach().cpu().numpy().tolist())
        tr_y.extend(y)

        # =========

        # train imdb
        batch_ = next(iter(train_imdb_data_iter))
        
        x,y=batch_
        loss, pred = model(x.to(device), y.to(device))
        aux_loss += loss.cpu().item()

        
        # backward
        loss = weighted*loss
        loss.backward()
        optimizer.step()
        model.zero_grad()

        aux_pred.extend(pred.detach().cpu().numpy().tolist())
        aux_y.extend(y)

        
    return (tr_loss/step, accuracy_score(tr_pred, tr_y)),(aux_loss/step, accuracy_score(aux_pred, aux_y))



def test_and_write(model, _data_iter, device,prefix,len_epo,out_file):
    preds= []

    for step, _batch in tqdm(enumerate(_data_iter)):
        model.eval()
        x,y=_batch
        with torch.no_grad():
            loss, pred = model(x.to(device), y.to(device))
            pred = pred.detach().cpu().numpy().tolist()
            preds.extend(pred)

    print(f'# {type} pred:', preds)

    with open(out_file,encoding="utf-8", mode='w') as f:
        for pred in preds:
            f.write("positive" if pred==1 else 'negative')
            f.write('\n')
