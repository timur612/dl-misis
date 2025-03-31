import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import pandas as pd

_OWNERSHIP_MAP = {
    'MORTGAGE': 0,
    'RENT': 1,
    'OWN': 2,
    'OTHER': 3
}

_INTENT_MAP = {
    'EDUCATION': 0,
    'HOMEIMPROVEMENT': 1,
    'MEDICAL': 2,
    'DEBTCONSOLIDATION': 3,
    'VENTURE': 4,
    'PERSONAL': 5
}

_LOAN_MAP = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3,
    'E': 4,
    'F': 5,
    'G': 6
}


class LoanDataset(Dataset):
    def __init__(self, data):
        self._data = data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        item = self._data.iloc[idx]
        return {
            'target': torch.scalar_tensor(item['loan_status'], dtype=torch.float32),
            'cat_features': {
                "person_home_ownership": torch.scalar_tensor(
                    _OWNERSHIP_MAP[item['person_home_ownership']], dtype=torch.long),
                "loan_intent": torch.scalar_tensor(_INTENT_MAP[item['loan_intent']],
                                                   dtype=torch.long),
                "loan_grade": torch.scalar_tensor(_LOAN_MAP[item['loan_grade']],
                                                  dtype=torch.long)
            },
            'numeric_features': {
                "person_age": torch.scalar_tensor(
                    item['person_age'] / 123,
                    dtype=torch.float32),
                "person_income": torch.scalar_tensor(
                    item['person_income'] / 1200000,
                    dtype=torch.float32),
                "loan_amnt": torch.scalar_tensor(
                    item['loan_amnt'] / 35000,
                    dtype=torch.float32),
                "person_emp_length": torch.scalar_tensor(
                    item['person_emp_length'] / 123,
                    dtype=torch.float32),
                "loan_int_rate": torch.scalar_tensor(
                    item['loan_int_rate'] / 23.22,
                    dtype=torch.float32),
                "loan_percent_income": torch.scalar_tensor(
                    item['loan_percent_income'] / 0.83, dtype=torch.float32)
            }

        }


class LoanCollator:
    def __call__(self, items):
        return {
            'target': torch.stack([x['target'] for x in items]),
            'cat_features': {
                "person_home_ownership": torch.stack(
                    [x['cat_features']['person_home_ownership'] for x in items]),
                "loan_intent": torch.stack(
                    [x['cat_features']['loan_intent'] for x in items]),
                "loan_grade": torch.stack(
                    [x['cat_features']['loan_grade'] for x in items]),
            },
            'numeric_features': {
                "person_age": torch.stack(
                    [x['numeric_features']['person_age'] for x in items]),
                "person_income": torch.stack(
                    [x['numeric_features']['person_income'] for x in items]),
                "loan_amnt": torch.stack(
                    [x['numeric_features']['loan_amnt'] for x in items]),
                "person_emp_length": torch.stack(
                    [x['numeric_features']['person_emp_length'] for x in items]),
                "loan_int_rate": torch.stack(
                    [x['numeric_features']['loan_int_rate'] for x in items]),
                "loan_percent_income": torch.stack(
                    [x['numeric_features']['loan_percent_income'] for x in items])
            }
        }


def load_dataset(train_file, test_file):
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)

    return LoanDataset(df_train), LoanDataset(df_test)

