from torch import nn
import torch

class MyRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))  # обучаемый масштаб

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt()  # RMS по последнему измерению
        x_norm = x / (rms + self.eps)  # нормализация
        return self.gamma * x_norm  # масштабируем


def test_rmsnorm():
    torch.manual_seed(42)
    x = torch.randn(2, 5, 10)  # batch=2, seq_len=5, dim=10

    my_norm = MyRMSNorm(dim=10)
    torch_norm = nn.RMSNorm(normalized_shape=10)

    # синхронизируем веса
    with torch.no_grad():
        torch_norm.weight.copy_(my_norm.gamma)

    out_my = my_norm(x)
    out_torch = torch_norm(x)

    print("Разница (L2 norm):", (out_my - out_torch).norm().item())
    print("Совпадают ли:", torch.allclose(out_my, out_torch, atol=1e-6))
    
if __name__ == "__main__":
    test_rmsnorm()
    # Разница (L2 norm): 1.3349517757887952e-06
    # Совпадают ли: True