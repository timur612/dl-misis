from datasets import load_dataset

def load_train_test(name: str = "sentence-transformers/natural-questions"):
    ds = load_dataset(name)
    ds_split = ds['train'].train_test_split(test_size=0.2, seed=42)
    train, test = ds_split['train'], ds_split['test']
    return train, test

def load_train_test_e5(name: str):
    ds = load_dataset(name)
    ds_split = ds['train'].train_test_split(test_size=0.2, seed=42)
    train, test = ds_split['train'], ds_split['test']
    def add_prefixes(example):
        example['query'] = "query: " + example['query']
        example['answer'] = "passage: " + example['answer']
        return example

    train = train.map(add_prefixes)
    test = test.map(add_prefixes)
    return train, test