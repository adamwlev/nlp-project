import torch
import torch.nn as nn
import numpy as np
from math import ceil
from collections import Counter
from transformers import AutoModel, AutoTokenizer
from eval import get_f1

data_path = 'data'

MODEL_NAME = "roberta-large"
EMBED_SIZE = 1024
HIDDEN_SIZE = 1568
BATCH_SIZE = 5
STEP_EVERY = 16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

label_map = {'O':0,'B':1,'I':2}
label_map_reverse = {0:'O',1:'B',2:'I'}

with open(f"{data_path}/train/train.txt") as f:
    train_lines = f.read().splitlines()

with open(f"{data_path}/wnut/train.txt") as f:
    train_lines += f.read().splitlines()

with open(f"{data_path}/dev/dev.txt") as f:
    dev_lines = f.read().splitlines()

with open(f"{data_path}/test/test.nolabels.txt") as f:
    test_lines = f.read().splitlines()

def get_counts(lines):
    seq = {"sub_tokens":[],"tokens":[]}
    MAX_NUM_SUB_TOK = 0
    MAX_NUM_TOK = 0
    for line in lines:
        if line:
            if '\t' in line:
                word, _ = line.split('\t')
            else:
                word = line
            if seq['tokens']:
                tokens = tokenizer.tokenize(' ' + word)
            else:
                tokens = tokenizer.tokenize(word)
            seq['sub_tokens'].extend(tokens)
            seq['tokens'].append(word)
        else:
            if len(seq['sub_tokens'])>MAX_NUM_SUB_TOK:
                MAX_NUM_SUB_TOK = len(seq['sub_tokens'])
            if len(seq['tokens'])>MAX_NUM_TOK:
                MAX_NUM_TOK = len(seq['tokens'])
            seq = {"sub_tokens":[],"tokens":[]}
    return MAX_NUM_SUB_TOK, MAX_NUM_TOK

MAX_NUM_SUB_TOK, MAX_NUM_TOK = get_counts(train_lines + dev_lines + test_lines)

def get_sequences(lines):
    to_ret = []
    seq = []
    for line in lines:
        if line:
            if '\t' in line:
                word, tag = line.split('\t')
            else:
                word, tag = line, 'O'
            tag = tag[0]
            seq.append((word,tag))
        else:
            to_ret.append(seq)
            seq = []
    return to_ret

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sub_tokens = []
        labels = []
        T = torch.zeros((MAX_NUM_TOK,MAX_NUM_SUB_TOK+2),dtype=torch.float)
        for word_ind, (word, tag) in enumerate(self.data[idx]):
            if not sub_tokens:
                sub_tokens_ = tokenizer.tokenize(word)
            else:
                sub_tokens_ = tokenizer.tokenize(' ' + word)
            
            start_sub_tok_ind = len(sub_tokens)
            for i in range(start_sub_tok_ind,start_sub_tok_ind+len(sub_tokens_)):
                T[word_ind,i+1] = 1.
            sub_tokens.extend(sub_tokens_)
            labels.append(label_map[tag])
        
        num_sub_toks = len(sub_tokens)
        sub_tokens = [tokenizer.bos_token] + sub_tokens + [tokenizer.sep_token]
        sub_tokens.extend([tokenizer.pad_token for _ in range(MAX_NUM_SUB_TOK-len(sub_tokens)+2)])
        sub_token_ids = torch.tensor(tokenizer.convert_tokens_to_ids(sub_tokens))
        attn_mask = torch.tensor([1 for _ in range(num_sub_toks+2)] +
                                 [0 for _ in range(MAX_NUM_SUB_TOK-num_sub_toks)])
        labels_mask = torch.tensor([1 for _ in labels] + [0 for _ in range(MAX_NUM_TOK-len(labels))]).bool()
        seq_len = torch.tensor(len(labels))
        labels = torch.tensor(labels + [-1 for _ in range(MAX_NUM_TOK-len(labels))])

        return {"sub_token_ids":sub_token_ids,
                "T":T,
                "attn_mask":attn_mask,
                "labels":labels,
                "seq_len":seq_len,
                "labels_mask":labels_mask}

class BertTagger(nn.Module):
    def __init__(self, emb_size, hid_size):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(MODEL_NAME)
        self.decoder = nn.LSTM(emb_size,hid_size,batch_first=True,bidirectional=False)
        self.dropout = nn.Dropout(p=.5)
        self.classifier = nn.Linear(emb_size, 3)
    
    def forward(self, x, attn_mask, T):
        ## x is batch X MAX_NUM_SUB_TOK+2
        ## attn_mask is the same
        ## T is batch X MAX_NUM_TOK X MAX_NUM_SUB_TOK+2
        outs = self.transformer(x,attention_mask=attn_mask)[0]
        outs = torch.bmm(T,outs)
        divisor = T.sum(2,keepdims=True)
        divisor[divisor==0] = 1e-5
        outs = torch.div(outs,divisor)
        outs, _ = self.decoder(outs)
        return self.classifier(self.dropout(outs))

def get_bert_tagger():
    model = BertTagger(EMBED_SIZE, HIDDEN_SIZE)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-6, weight_decay=1e-3)
    return model, criterion, optimizer

def decode(preds,trues,seq_lens):
    ## preds and true should be 1d numpy array of ints
    ## seq_lens should be 1d numpy array of ints
    global label_map_reverse
    preds_out = []
    trues_out = []
    i = 0
    for seq_len in seq_lens:
        preds_out.append([label_map_reverse[pred] for pred in preds[i:i+seq_len]])
        trues_out.append([label_map_reverse[true] for true in trues[i:i+seq_len]])
        i += seq_len
    return preds_out, trues_out

def train_tagger(model_getter, data, batch_size=BATCH_SIZE, step_every=STEP_EVERY, val_frac=.07, tol=2):

    global device

    model, criterion, optimizer = model_getter()

    train = data
    val_inds = set(np.random.choice(list(range(len(train))),replace=False,
                                    size=int(val_frac*len(train))).tolist())
    val = []
    for ind in val_inds:
        val.append(train[ind])
    train = [ex for i,ex in enumerate(train) if i not in val_inds]

    train_ds = Dataset(train)
    train_dl = torch.utils.data.DataLoader(train_ds,batch_size=batch_size,shuffle=True)

    val_ds = Dataset(val)
    val_dl = torch.utils.data.DataLoader(val_ds,batch_size=batch_size,shuffle=False)

    best_val_f1, best_epoch = -np.inf, 0
    epoch = 0
    while True:
        epoch += 1
        model.train()

        train_pred_scores = torch.empty((0,),dtype=torch.float)
        train_preds = np.empty((0,),dtype=np.int64)
        train_labels = np.empty((0,),dtype=np.int64)
        train_seq_lens = np.empty((0,),dtype=np.int64)
        
        val_pred_scores = torch.empty((0,),dtype=torch.float)
        val_preds = np.empty((0,),dtype=np.int64)
        val_labels = np.empty((0,),dtype=np.int64)
        val_seq_lens = np.empty((0,),dtype=np.int64)

        for i, batch in enumerate(train_dl):
            bs = batch['sub_token_ids'].size(0)
            sub_token_ids = batch['sub_token_ids'].to(device)
            attn_mask = batch['attn_mask'].to(device)
            T = batch['T'].to(device)
            labels = batch['labels'].view(bs*MAX_NUM_TOK).to(device)
            labels_mask = batch['labels_mask'].view(bs*MAX_NUM_TOK).to(device)

            out = model(sub_token_ids,attn_mask,T)
            preds = out.view(bs*MAX_NUM_TOK,3)
            loss = criterion(preds[labels_mask],labels[labels_mask])
            loss.backward()

            if (i+1)%step_every==0:
                optimizer.step()
                optimizer.zero_grad()
            
            preds = preds[labels_mask].cpu().detach()
            labels = labels[labels_mask].cpu().numpy()
            train_pred_scores = torch.cat([train_pred_scores,preds])
            train_preds = np.concatenate([train_preds,preds.argmax(1).numpy()])
            train_labels = np.concatenate([train_labels,labels])
            train_seq_lens = np.concatenate([train_seq_lens,batch['seq_len']])
            
        model.eval()

        with torch.set_grad_enabled(False):
            for batch in val_dl:
                bs = batch['sub_token_ids'].size(0)
                sub_token_ids = batch['sub_token_ids'].to(device)
                attn_mask = batch['attn_mask'].to(device)
                T = batch['T'].to(device)
                labels = batch['labels'].view(bs*MAX_NUM_TOK).to(device)
                labels_mask = batch['labels_mask'].view(bs*MAX_NUM_TOK).to(device)

                out = model(sub_token_ids,attn_mask,T)
                preds = out.view(bs*MAX_NUM_TOK,3)
                
                preds = preds[labels_mask].cpu().detach()
                labels = labels[labels_mask].cpu().numpy()
                val_pred_scores = torch.cat([val_pred_scores,preds])
                val_preds = np.concatenate([val_preds,preds.argmax(1).numpy()])
                val_labels = np.concatenate([val_labels,labels])
                val_seq_lens = np.concatenate([val_seq_lens,batch['seq_len']])

        train_loss = criterion(train_pred_scores,
                               torch.from_numpy(train_labels)).item()
        val_loss = criterion(val_pred_scores,
                             torch.from_numpy(val_labels)).item()
        
        train_acc = (train_preds==train_labels).mean()
        val_acc = (val_preds==val_labels).mean()

        train_pred_seqs, train_true_seqs = decode(train_preds,train_labels,train_seq_lens)
        train_f1 = get_f1(train_pred_seqs,train_true_seqs)
        val_pred_seqs, val_true_seqs = decode(val_preds,val_labels,val_seq_lens)
        val_f1 = get_f1(val_pred_seqs,val_true_seqs)

        print(f'epoch={epoch}, train loss={train_loss:.4f}, '
              f'val loss={val_loss:.4f}, train acc={train_acc:.4f}, '
              f'val acc={val_acc:.4f}, train_f1={train_f1:.4f}, val_f1={val_f1:.4f}')

        if val_f1>best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
        elif epoch>tol and epoch>best_epoch+tol:
            print('Early stopping triggered')
            break

    ds = Dataset(data)
    dl = torch.utils.data.DataLoader(ds,batch_size=batch_size,shuffle=True)

    print('retraining')
    model, criterion, optimizer = model_getter()
    for epoch in range(best_epoch):
        for i, batch in enumerate(dl):
            bs = batch['sub_token_ids'].size(0)
            sub_token_ids = batch['sub_token_ids'].to(device)
            attn_mask = batch['attn_mask'].to(device)
            T = batch['T'].to(device)
            labels = batch['labels'].view(bs*MAX_NUM_TOK).to(device)
            labels_mask = batch['labels_mask'].view(bs*MAX_NUM_TOK).to(device)

            out = model(sub_token_ids,attn_mask,T)
            preds = out.view(bs*MAX_NUM_TOK,3)
            loss = criterion(preds[labels_mask],labels[labels_mask])
            loss.backward()

            if (i+1)%step_every==0:
                optimizer.step()
                optimizer.zero_grad()

    return model

def save_preds(taggers, seqs, path):
    ds = Dataset(seqs)
    dl = torch.utils.data.DataLoader(ds,batch_size=1,shuffle=False)
    seq_preds = [[] for _ in range(len(taggers))]

    for i in range(len(taggers)):
        taggers[i].eval()
        for batch in dl:
            sub_token_ids = batch['sub_token_ids'].to(device)
            attn_mask = batch['attn_mask'].to(device)
            T = batch['T'].to(device)
            labels_mask = batch['labels_mask'].view(MAX_NUM_TOK).to(device)
            with torch.no_grad():
                out = taggers[i](sub_token_ids,attn_mask,T)
                preds = out.view(MAX_NUM_TOK,3)
                preds = preds[labels_mask].cpu().detach()
                preds = preds.argmax(1).numpy()
                seq_preds[i].append([label_map_reverse[p] for p in preds])
        
    with open(path,'w') as f:
        for seq_preds_ in zip(*seq_preds):
            in_span = False
            for ps in zip(*seq_preds_):
                c = Counter(ps)
                p = max(c,key=c.get)
                if p=='I' and not in_span:
                    p = 'B'
                f.write(p)
                f.write('\n')
                in_span = p=='B' or p=='I'
            f.write('\n')

def train_and_predict(tr_lines, pr_lines, path, num_taggers=1):
    taggers = []
    for i in range(num_taggers):
        tagger = train_tagger(get_bert_tagger, get_sequences(tr_lines))
        taggers.append(tagger)
    save_preds(taggers, get_sequences(pr_lines), path)

#train_and_predict(train_lines, dev_lines, 'dev.out')
train_and_predict(train_lines+dev_lines, test_lines, 'test.out')

