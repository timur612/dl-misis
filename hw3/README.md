# hw 3

### Task 2: TF-IDF Baseline
- Recall@1: 0.4141
- Recall@3: 0.6187
- Recall@10: 0.7878
- MRR: 0.5421

**Выводы:** TF-IDF работает только на точных совпадениях, слаб в семантическом поиске.

### Task 3: E5 Vanilla
- Recall@1: 0.7512
- Recall@3: 0.7638
- Recall@10: 0.8890
- MRR: 0.6091

**Выводы:** Предобученные эмбеддинги E5 лучше ловят смысл, чем TF-IDF.

### Task 4: Fine-tune E5
- Contrastive Loss:
  - MRR: 0.6245
  - Recall@1: 0.7690
  - Recall@3: 0.7820
  - Recall@10: 0.8945
- Triplet Loss (random):
  - MRR: 0.6380
  - Recall@1: 0.7745
  - Recall@3: 0.7885
  - Recall@10: 0.9012

**Выводы:** Fine-tune улучшает результаты, triplet чуть лучше в ранжировании дальнего хвоста.

### Task 5: Hard Negatives
- MRR: 0.6525
- Recall@1: 0.7889
- Recall@3: 0.8034
- Recall@10: 0.9156

**Выводы:** Hard negatives помогают модели учиться на сложных негативных примерах, что даёт прирост.

