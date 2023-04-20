import numpy as np
import torch

from pytorch_ops import sharp_sdtw_div_value_and_grad
from numba_ops import sharp_sdtw_div_value_and_grad as baseline

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    X = torch.randn(3, 100).to(device)
    y = torch.randn(3, 10).to(device)
    
    model = torch.nn.Linear(100, 10).to(device)
    
    x = model(X)
    
    value, grad = sharp_sdtw_div_value_and_grad(x, y)
    
    x.backward(grad)
    
    assert model.weight.grad is not None
    
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    baseline_value, baseline_grad = baseline(x, y)
    
    value = value.detach().cpu().numpy()
    grad = grad.detach().cpu().numpy()
    
    assert np.allclose(value, baseline_value, 1e-4, 1e-3)
    print(np.abs((value - baseline_value)).max())

    assert np.allclose(grad, baseline_grad, 1e-4, 1e-3)
    print(np.abs((grad - baseline_grad)).max())