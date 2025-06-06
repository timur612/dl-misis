import random
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import numpy as np
import argparse

class ContrastiveDataset(Dataset):
    def __init__(self, questions, docs, tokenizer, max_len=256):
        self.q = questions
        self.d = docs
        self.tk = tokenizer
        self.max_len = max_len
    def __len__(self): return len(self.q)
    def __getitem__(self, i):
        q = self.q[i]; pos = self.d[i]; neg = random.choice(self.d)
        def tok(x):
            return self.tk(x, truncation=True, padding='max_length',
                           max_length=self.max_len, return_tensors='pt')
        return tok(q), tok(pos), tok(neg)

class TripletDataset(Dataset):
    def __init__(self, questions, docs, tokenizer, max_len=256):
        self.q = questions
        self.d = docs
        self.tk = tokenizer
        self.max_len = max_len
    def __len__(self): return len(self.q)
    def __getitem__(self, i):
        q = self.q[i]; pos = self.d[i]; neg = random.choice(self.d)
        def tok(x):
            return self.tk(x, truncation=True, padding='max_length',
                           max_length=self.max_len, return_tensors='pt')
        return tok(q), tok(pos), tok(neg)

def train_contrastive(model, loader, optimizer, device, epochs=1):
    cos = nn.CosineSimilarity(dim=1)
    model.train()
    for _ in range(epochs):
        for q, pos, neg in loader:
            # move inputs to device
            for dct in (q, pos, neg):
                for k,v in dct.items(): dct[k]=v.squeeze(1).to(device)
            q_emb = model(**q).pooler_output
            pos_emb = model(**pos).pooler_output
            neg_emb = model(**neg).pooler_output
            sim_pos = cos(q_emb, pos_emb)
            sim_neg = cos(q_emb, neg_emb)
            loss = -torch.log(torch.sigmoid(sim_pos - sim_neg)).mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()

def train_triplet(model, loader, optimizer, device, epochs=1, margin=0.2):
    triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=lambda x,y:1 - nn.functional.cosine_similarity(x,y), margin=margin)
    model.train()
    for _ in range(epochs):
        for q, pos, neg in loader:
            for dct in (q, pos, neg):
                for k,v in dct.items(): dct[k]=v.squeeze(1).to(device)
            q_emb = model(**q).pooler_output
            pos_emb = model(**pos).pooler_output
            neg_emb = model(**neg).pooler_output
            loss = triplet_loss(q_emb, pos_emb, neg_emb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

def encode_embeddings(model, texts, tokenizer, device, batch_size=32):
    loader = DataLoader(texts, batch_size=batch_size, shuffle=False)
    embs = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            toks = tokenizer(batch, truncation=True, padding='longest', return_tensors='pt').to(device)
            out = model(**toks).pooler_output
            embs.append(out.cpu().numpy())
    return np.vstack(embs)

def evaluate(q_emb, d_emb, q2d_idx):
    sims = q_emb @ d_emb.T
    ranks = np.argsort(-sims, axis=1)
    mrr = 0.; recall = {1:0, 3:0, 10:0}
    for i, r in enumerate(ranks):
        rank = np.where(r == q2d_idx[i])[0][0] + 1
        mrr += 1/r
        for k in recall:
            if rank <= k: recall[k] += 1
    n = len(q_emb)
    print(f"MRR: {mrr/n:.4f}")
    for k in recall: print(f"Recall@{k}: {recall[k]/n:.4f}")

def run_contrastive(qs_train, ds_train, tokenizer, device, epochs=1, lr=3e-5):
    """Train Multilingual E5 model with contrastive loss."""
    model = AutoModel.from_pretrained('multilingual-e5-base').to(device)
    loader = DataLoader(ContrastiveDataset(qs_train, ds_train, tokenizer), batch_size=16, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_contrastive(model, loader, optimizer, device, epochs)
    return model


def run_triplet(qs_train, ds_train, tokenizer, device, epochs=1, lr=3e-5):
    """Train Multilingual E5 model with random negative triplet loss."""
    model = AutoModel.from_pretrained('multilingual-e5-base').to(device)
    loader = DataLoader(TripletDataset(qs_train, ds_train, tokenizer), batch_size=16, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_triplet(model, loader, optimizer, device, epochs)
    return model


def run_hard_triplet(qs_train, ds_train, tokenizer, device, epochs=1, lr=3e-5):
    """Train Multilingual E5 model with hard negatives for triplet loss."""
    # начальная модель и эмбеддинги для выбора hard negatives
    base_model = AutoModel.from_pretrained('multilingual-e5-base').to(device)
    train_q_emb = encode_embeddings(base_model, qs_train, tokenizer, device)
    train_d_emb = encode_embeddings(base_model, ds_train, tokenizer, device)
    sims = train_q_emb @ train_d_emb.T
    hard_idxs = []
    for i in range(len(qs_train)):
        sims_i = sims[i].copy(); sims_i[i] = -1
        hard_idxs.append(sims_i.argmax())
    # подготовка датасета и обучение
    class HardTripletDataset(Dataset):
        def __init__(self, questions, docs, hard_idxs, tokenizer, max_len=256):
            self.q, self.d, self.h, self.tk, self.max_len = questions, docs, hard_idxs, tokenizer, max_len
        def __len__(self): return len(self.q)
        def __getitem__(self, idx):
            q, pos, neg = self.q[idx], self.d[idx], self.d[self.h[idx]]
            def tok(x): return self.tk(x, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
            return tok(q), tok(pos), tok(neg)
    model = AutoModel.from_pretrained('multilingual-e5-base').to(device)
    loader = DataLoader(HardTripletDataset(qs_train, ds_train, hard_idxs, tokenizer), batch_size=16, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_triplet(model, loader, optimizer, device, epochs)
    return model

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate E5 retrieval models')
    parser.add_argument('--mode', type=str, choices=['contrastive', 'triplet', 'hard_triplet', 'all'], required=True,
                        help='Выбрать способ обучения или all для всех')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = load_dataset('your_dataset') 
    train = ds['train']; test = ds['test']
    qs_train, ds_train = train['question'], train['document']
    qs_test, ds_test = test['question'], test['document']

    tokenizer = AutoTokenizer.from_pretrained('multilingual-e5-base')
    # Обучение выбранных моделей
    if args.mode in ['contrastive', 'all']:
        print('=== Обучение Contrastive ===')
        model_c = run_contrastive(qs_train, ds_train, tokenizer, device)
    if args.mode in ['triplet', 'all']:
        print('=== Обучение Triplet (random negatives) ===')
        model_t = run_triplet(qs_train, ds_train, tokenizer, device)
    if args.mode in ['hard_triplet', 'all']:
        print('=== Обучение Triplet (hard negatives) ===')
        model_h = run_hard_triplet(qs_train, ds_train, tokenizer, device)

    # Оценка моделей
    if 'contrastive' in args.mode or args.mode == 'all':
        print('--- Результаты Contrastive ---')
        q_emb = encode_embeddings(model_c, qs_test, tokenizer, device)
        d_emb = encode_embeddings(model_c, ds_test, tokenizer, device)
        evaluate(q_emb, d_emb, list(range(len(qs_test))))
    if 'triplet' in args.mode or args.mode == 'all':
        print('--- Результаты Triplet (random negatives) ---')
        q_emb = encode_embeddings(model_t, qs_test, tokenizer, device)
        d_emb = encode_embeddings(model_t, ds_test, tokenizer, device)
        evaluate(q_emb, d_emb, list(range(len(qs_test))))
    if 'hard_triplet' in args.mode or args.mode == 'all':
        print('--- Результаты Triplet (hard negatives) ---')
        q_emb = encode_embeddings(model_h, qs_test, tokenizer, device)
        d_emb = encode_embeddings(model_h, ds_test, tokenizer, device)
        evaluate(q_emb, d_emb, list(range(len(qs_test))))

if __name__ == '__main__':
    main()