<img src="./adan-pseudocode.png" width="450px"></img>

## Adan - Pytorch

Implementation of the <a href="https://arxiv.org/abs/2208.06677">Adan</a> (ADAptive Nesterov momentum algorithm) Optimizer in Pytorch.

Explanation from <a href="https://twitter.com/davisblalock/status/1561976182567870465">Davis Blalock</a>

<a href="https://github.com/sail-sg/Adan">Official Adan code</a>

## Install

```bash
$ pip install adan-pytorch
```

## Usage

```python
from adan_pytorch import Adan

# mock model

import torch
from torch import nn

model = torch.nn.Sequential(
    nn.Linear(16, 16),
    nn.GELU()
)

# instantiate Adan with model parameters

optim = Adan(
    model.parameters(),
    lr = 1e-3,                  # learning rate (can be much higher than Adam, up to 5-10x)
    betas = (0.02, 0.08, 0.01), # beta 1-2-3 as described in paper - author says most sensitive to beta3 tuning
    weight_decay = 0.02         # weight decay 0.02 is optimal per author
)

# train

for _ in range(10):
    loss = model(torch.randn(16)).sum()
    loss.backward()
    optim.step()
    optim.zero_grad()

```

## Citations

```bibtex
@article{Xie2022AdanAN,
    title   = {Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models},
    author  = {Xingyu Xie and Pan Zhou and Huan Li and Zhouchen Lin and Shuicheng Yan},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2208.06677}
}
```
