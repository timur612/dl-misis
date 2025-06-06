import torch
from torch.autograd import Function
from typing import Tuple, Any

class ExpCosFunction(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Сохраняем входы для backward
        ctx.save_for_backward(x, y)
        return torch.exp(x) + torch.cos(y)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = ctx.saved_tensors

        dx = torch.exp(x) * grad_output       # ∂f/∂x = e^x
        dy = -torch.sin(y) * grad_output      # ∂f/∂y = -sin(y)

        return dx, dy
    
def test_exp_cos_autograd():
    torch.manual_seed(42)

    # Входы
    x = torch.randn(4, 5, requires_grad=True)
    y = torch.randn(4, 5, requires_grad=True)

    # Обычная функция
    fx = torch.exp(x) + torch.cos(y)
    loss1 = fx.sum()
    loss1.backward()
    grad_x_ref = x.grad.clone()
    grad_y_ref = y.grad.clone()

    # Обнуляем градиенты
    x.grad.zero_()
    y.grad.zero_()

    # Кастомная функция
    fx_custom = ExpCosFunction.apply(x, y)
    loss2 = fx_custom.sum()
    loss2.backward()
    grad_x_custom = x.grad
    grad_y_custom = y.grad

    # Проверки
    print("dx совпадает:", torch.allclose(grad_x_ref, grad_x_custom, atol=1e-6))
    print("dy совпадает:", torch.allclose(grad_y_ref, grad_y_custom, atol=1e-6))
    print("Разница L2 (dx):", (grad_x_ref - grad_x_custom).norm().item())
    print("Разница L2 (dy):", (grad_y_ref - grad_y_custom).norm().item())

if __name__ == "__main__":
    test_exp_cos_autograd()
    # dx совпадает: True
    # dy совпадает: True
    # Разница L2 (dx): 0.0
    # Разница L2 (dy): 0.0