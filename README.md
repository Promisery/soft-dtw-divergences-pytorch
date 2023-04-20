Differentiable Divergences between Time Series
==============================================

A pytorch implementation of soft-DTW divergences based on [soft-dtw-divergences](https://github.com/google-research/soft-dtw-divergences).

Example
-------

```python
import torch
from pytorch_ops import sharp_sdtw_div_value_and_grad

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
X = torch.randn(3, 100).to(device)
y = torch.randn(3, 10).to(device)

model = torch.nn.Linear(100, 10).to(device)

x = model(X)

value, grad = sharp_sdtw_div_value_and_grad(x, y)

x.backward(grad)
```

Reference
----------

> Differentiable Divergences between Time Series <br/>
> Mathieu Blondel, Arthur Mensch, Jean-Philippe Vert <br/>
> [arXiv:2010.08354](https://arxiv.org/abs/2010.08354)