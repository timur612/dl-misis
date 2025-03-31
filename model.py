import torch
from torch import nn


class BaseBlock(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(hidden_size)
        self.linear_1 = nn.Linear(hidden_size, hidden_size * 4)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.linear_2 = nn.Linear(hidden_size * 4, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(x)
        x = self.linear_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        return x


class LoanModel(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.person_home_ownership = nn.Embedding(4, hidden_size)
        self.loan_intent = nn.Embedding(6, hidden_size)
        self.loan_grade = nn.Embedding(7, hidden_size)

        self.numeric_linear = nn.Linear(6, hidden_size)

        self.block1 = BaseBlock(hidden_size)
        self.block2 = BaseBlock(hidden_size)
        self.block3 = BaseBlock(hidden_size)

        self.out = nn.Linear(hidden_size, 1)

    def forward(self, cat_features, numeric_features):
        x_ownership = self.person_home_ownership(cat_features['person_home_ownership'])
        x_intent = self.loan_intent(cat_features['loan_intent'])
        x_grade = self.loan_grade(cat_features['loan_grade'])

        stacked_numeric = torch.stack(
            [numeric_features['person_age'], numeric_features['person_income'],
             numeric_features['loan_amnt'], numeric_features['person_emp_length'],
             numeric_features['loan_int_rate'],
             numeric_features['loan_percent_income']], dim=1)
        x_numeric = self.numeric_linear(stacked_numeric)

        x_total = x_ownership + x_intent + x_grade + x_numeric

        x_total = self.block1(x_total) + x_total
        x_total = self.block2(x_total) + x_total
        x_total = self.block3(x_total) + x_total
        x = self.out(x_total)

        return x.squeeze(-1)
