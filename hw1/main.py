import torch
from train import Trainer

if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    """Эксперимент 1. Простая модель"""
    # trainer = Trainer(["data/loan_train.csv", "data/loan_test.csv"],
    #                   num_epochs=50,
    #                   lr=0.01,
    #                   batch_size=32,
    #                   hidden_size=32,
    #                   device=device)
    # trainer.run()
    # оптимальное количество эпох - 9
    # после 9 эпохи, видим, что eval_loss растет, значит надо заканчивать обучение

    """Эксперимент 2. Модель побольше"""
    # trainer = Trainer(["data/loan_train.csv", "data/loan_test.csv"],
    #                   num_epochs=50,
    #                   lr=0.01,
    #                   batch_size=32,
    #                   hidden_size=128,
    #                   device=device)
    # trainer.run()
    # стало учиться быстрее, хватает 5 эпох, для того, чтобы получить метрику 0.9
    # после 5 эпохи сильно переобучается

    """Эксперимент 3. Skip Connections, Batch Norms"""
    trainer = Trainer(["data/loan_train.csv", "data/loan_test.csv"],
                      num_epochs=50,
                      lr=0.01,
                      batch_size=32,
                      hidden_size=128,
                      device=device)
    trainer.run()
    # стало учиться еще быстрее, хватает 3 эпох, но метрика остается +- такой же
    # после 3 эпохи сильно переобучается

    """Эксперимент 4. Dropout"""
    # trainer = Trainer("data/loan_train.csv",
    #                   num_epochs=10,
    #                   lr=0.01,
    #                   batch_size=32,
    #                   hidden_size=128,
    #                   device=device)
    # trainer.run()
    # пробовал dropoout 0.01, 0.2 и 0.9. при 0.01 быстро переобучается, при 0.9 недообучается.
    # с 0.2 после 2 эпохи идет переобучения и метрика растет

    """Эксперимент 5. Weight Decay, Learning Rate"""
    # trainer = Trainer(["data/loan_train.csv", "data/loan_test.csv"],
    #                   num_epochs=10,
    #                   lr=0.01,
    #                   weight_decay=0.01,
    #                   batch_size=32,
    #                   hidden_size=128,
    #                   device=device)
    # trainer.run()

    # weight_decay=0.1 плохо
    # weight_decay=0.01 хорошо