from tqdm import tqdm
import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import SGD
from torchmetrics import MeanMetric, AUROC
from torch.utils.data import DataLoader
import wandb

from model import LoanModel
from dataset import load_dataset, LoanCollator


class Trainer:
    def __init__(self, dataset_files, num_epochs=100, lr=0.01, weight_decay=0.1,
                 batch_size=32, hidden_size=32, device="cpu"):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device

        self.model = LoanModel(hidden_size=hidden_size)
        self.model.to(device)
        self.loss = BCEWithLogitsLoss()

        self.optimizer = SGD(self.model.parameters(),
                             lr=lr,
                             # weight_decay=weight_decay
                             )

        self.collator = LoanCollator()
        self.train_dataset, self.test_dataset = load_dataset(dataset_files[0],
                                                             dataset_files[1])

        self.run_wandb = wandb.init(
            entity="m2207772",
            project="hw1",
            config={
                "learning_rate": lr,
                "architecture": "TableNN",
                "dataset": "LoanDataset",
                "epochs": self.num_epochs,
            },
        )

    def run(self):
        train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                      collate_fn=self.collator)
        eval_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size,
                                     collate_fn=self.collator)

        for epoch in range(self.num_epochs):
            train_loss, train_rocauc = self._train(train_dataloader, epoch + 1)
            eval_loss, eval_rocauc = self._eval(eval_dataloader, epoch + 1)
            self.run_wandb.log({"train_loss": train_loss, "train_rocauc": train_rocauc,
                                "eval_loss": eval_loss, "eval_rocauc": eval_rocauc})

        self.run_wandb.finish()
        return self.model

    def _train(self, train_dataloader, epoch):
        train_loss = MeanMetric().to(self.device)
        train_rocauc = AUROC(task='binary').to(self.device)

        self.model.train()
        for batch in tqdm(train_dataloader):
            self._map_to_device(batch, self.device)
            target, cat_features, numeric_features = batch['target'], batch[
                'cat_features'], batch['numeric_features']
            prediction = self.model(cat_features, numeric_features)
            loss_value = self.loss(prediction, target)
            loss_value.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            train_loss.update(loss_value)
            train_rocauc.update(torch.sigmoid(prediction), target)

        train_loss = train_loss.compute().item()
        train_rocauc = train_rocauc.compute().item()
        # print( f"Epoch: {epoch}/{self.num_epochs}\nTrain Loss: {train_loss}\nTrain
        # ROC-AUC: {train_rocauc}")
        return train_loss, train_rocauc

    def _eval(self, val_dataloader, epoch):
        eval_loss = MeanMetric().to(self.device)
        eval_rocauc = AUROC(task='binary').to(self.device)

        self.model.eval()
        for batch in tqdm(val_dataloader):
            self._map_to_device(batch, self.device)
            target, cat_features, numeric_features = batch['target'], batch[
                'cat_features'], batch['numeric_features']
            with torch.no_grad():
                prediction_eval = self.model(cat_features, numeric_features)
                eval_loss_value = self.loss(prediction_eval, target)

            eval_loss.update(eval_loss_value)
            eval_rocauc.update(torch.sigmoid(prediction_eval), target)
        eval_loss = eval_loss.compute().item()
        eval_rocauc = eval_rocauc.compute().item()
        # print( f"Epoch: {epoch}/{self.num_epochs}\nEval Loss: {eval_loss}\nEval
        # ROC-AUC: {eval_rocauc}")
        return eval_loss, eval_rocauc

    def _map_to_device(self, batch, dev):
        batch['target'] = batch['target'].to(dev)

        batch['cat_features']['person_home_ownership'] = batch['cat_features'][
            'person_home_ownership'].to(dev)
        batch['cat_features']['loan_intent'] = batch['cat_features']['loan_intent'].to(
            dev)
        batch['cat_features']['loan_grade'] = batch['cat_features']['loan_grade'].to(
            dev)

        batch['numeric_features']['person_age'] = batch['numeric_features'][
            'person_age'].to(dev)
        batch['numeric_features']['person_income'] = batch['numeric_features'][
            'person_income'].to(dev)
        batch['numeric_features']['loan_amnt'] = batch['numeric_features'][
            'loan_amnt'].to(dev)
        batch['numeric_features']['person_emp_length'] = batch['numeric_features'][
            'person_emp_length'].to(
            dev)
        batch['numeric_features']['loan_int_rate'] = batch['numeric_features'][
            'loan_int_rate'].to(
            dev)
        batch['numeric_features']['loan_percent_income'] = batch['numeric_features'][
            'loan_percent_income'].to(
            dev)
