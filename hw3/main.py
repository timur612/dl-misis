from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from torch import Tensor, mm
import torch

from dataset import load_train_test, load_train_test_e5
from tfidf_baseline import make_tfidf_vectors
from metrics import recall_topk, mean_reciprocal_rank as MRR

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def tfidf_vectors(train, test):
    # Recall@1: 0.4141
    # Recall@3: 0.6187
    # Recall@10: 0.7878
    # MRR: 0.5421
    _, vectorizer = make_tfidf_vectors(train['query'] + train['answer'])
    test_question_vecs = vectorizer.transform(test['query'])
    test_answer_vecs = vectorizer.transform(test['answer'])

    sims = cosine_similarity(test_question_vecs, test_answer_vecs)

    print(sims.shape)
    ranked_indices = np.argsort(-sims, axis=1)
    true_indices = np.arange(sims.shape[0])
    
    recall_1 = recall_topk(ranked_indices, true_indices, k=1)
    recall_3 = recall_topk(ranked_indices, true_indices, k=3)
    recall_10 = recall_topk(ranked_indices, true_indices, k=10)
    mrr = MRR(ranked_indices, true_indices)

    print(f"Recall@1: {recall_1:.4f}")
    print(f"Recall@3: {recall_3:.4f}")
    print(f"Recall@10: {recall_10:.4f}")
    print(f"MRR: {mrr:.4f}")

def e5_vectors(train, test):
    # Recall@1: 0.7512
    # Recall@3: 0.7638
    # Recall@10: 0.8890
    # MRR: 0.6091
    tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-base')
    model = AutoModel.from_pretrained('intfloat/multilingual-e5-base')
    batch_size = 32

    # move model to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    def encode(texts: list[str]) -> Tensor:
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt').to(device)
            outputs = model(**inputs)
            emb = average_pool(outputs.last_hidden_state, inputs['attention_mask'])
            embeddings.append(emb.detach().cpu())
        return torch.cat(embeddings, dim=0)

    # prepare data
    test_queries = test['query']
    test_answers = test['answer']

    # encode and normalize
    q_embs = F.normalize(encode(test_queries), p=2, dim=1)
    a_embs = F.normalize(encode(test_answers), p=2, dim=1)

    # cosine similarities and metrics
    sims = q_embs @ a_embs.T  # [N, N]
    sims_np = sims.numpy()
    ranked = np.argsort(-sims_np, axis=1)
    true_idx = np.arange(sims_np.shape[0])

    r1 = recall_topk(ranked, true_idx, k=1)
    r3 = recall_topk(ranked, true_idx, k=3)
    r10 = recall_topk(ranked, true_idx, k=10)
    mrr = MRR(ranked, true_idx)

    print(f"Recall@1: {r1:.4f}")
    print(f"Recall@3: {r3:.4f}")
    print(f"Recall@10: {r10:.4f}")
    print(f"MRR: {mrr:.4f}")

if __name__ == "__main__":
    # train, test = load_train_test(name="sentence-transformers/natural-questions")
    train, test = load_train_test_e5(name="sentence-transformers/natural-questions")

    e5_vectors(train, test)