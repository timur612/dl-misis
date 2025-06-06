from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from torch import Tensor, mm

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
    tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-base')
    model = AutoModel.from_pretrained('intfloat/multilingual-e5-base')

    test_tokeinzed_query = tokenizer(test['query'], max_length=512, padding=True, truncation=True, return_tensors='pt')
    test_tokeinzed_answer= tokenizer(test['answer'], max_length=512, padding=True, truncation=True, return_tensors='pt')
    outputs_query = model(**test_tokeinzed_query)
    embeddings_query = average_pool(outputs_query.last_hidden_state, test_tokeinzed_query['attention_mask'])
    test_question_vecs = F.normalize(embeddings_query, p=2, dim=1)
    outputs_answer = model(**test_tokeinzed_answer)
    embeddings_answer = average_pool(outputs_answer.last_hidden_state, test_tokeinzed_answer['attention_mask'])
    test_answer_vecs = F.normalize(embeddings_answer, p=2, dim=1)

    # Compute cosine similarity
    sims = (test_question_vecs @ test_answer_vecs.T)

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


if __name__ == "__main__":
    # train, test = load_train_test(name="sentence-transformers/natural-questions")
    train, test = load_train_test_e5(name="sentence-transformers/natural-questions")

    e5_vectors(train, test)