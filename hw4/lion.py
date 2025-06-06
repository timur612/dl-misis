import torch
import torch.nn as nn
from torch.optim import Optimizer

class Lion(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.99),
        weight_decay=0.0,
    ):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super(Lion, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if 'exp_avg' not in state:
                    state['exp_avg'] = torch.zeros_like(p.data)

                exp_avg = state['exp_avg']

                update = (1 - beta1) * grad + beta1 * exp_avg
                update = torch.sign(update)

                if weight_decay != 0:
                    update = update + weight_decay * p.data

                p.data.add_(update, alpha=-lr)

                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

def test_lion_optimizer():
    torch.manual_seed(0)

    model = nn.Sequential(
        nn.Linear(10, 1)
    ).to('cuda' if torch.cuda.is_available() else 'cpu')

    x = torch.randn(100, 10).to(model[0].weight.device)
    y = torch.randn(100, 1).to(model[0].weight.device)

    criterion = nn.MSELoss()
    optimizer = Lion(model.parameters(), lr=1e-3)

    for epoch in range(100):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

if __name__ == "__main__":
    test_lion_optimizer()
    # Epoch 0 | Loss: 1.0317
    # Epoch 10 | Loss: 1.0078
    # Epoch 20 | Loss: 0.9856
    # Epoch 30 | Loss: 0.9651
    # Epoch 40 | Loss: 0.9463
    # Epoch 50 | Loss: 0.9291
    # Epoch 60 | Loss: 0.9135
    # Epoch 70 | Loss: 0.8997
    # Epoch 80 | Loss: 0.8875
    # Epoch 90 | Loss: 0.8770