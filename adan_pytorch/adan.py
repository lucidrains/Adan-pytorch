import torch
from torch.optim import Optimizer

def exists(val):
    return val is not None

class Adan(Optimizer):
    def __init__(
        self,
        params,
        lr = 1e-3,
        betas = (0.02, 0.08, 0.01),
        eps = 1e-8,
        weight_decay = 0
    ):
        assert len(betas) == 3

        defaults = dict(
            lr = lr,
            betas = betas,
            eps = eps,
            weight_decay = weight_decay
        )

        super().__init__(params, defaults)

    def step(self, closure = None):
        loss = None

        if exists(closure):
            loss = closure()

        for group in self.param_groups:

            lr = group['lr']
            beta1, beta2, beta3 = group['betas']
            weight_decay = group['weight_decay']
            eps = group['eps']

            for p in group['params']:
                if not exists(p.grad):
                    continue

                data, grad = p.data, p.grad.data
                assert not grad.is_sparse

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(grad)
                    state['v'] = torch.zeros_like(grad)
                    state['n'] = torch.zeros_like(grad)

                step, m, v, n = state['step'], state['m'], state['v'], state['n']

                if step > 0:
                    prev_grad = state['prev_grad']

                    # main algorithm

                    m.mul_(1 - beta1).add_(grad, alpha = beta1)

                    grad_diff = grad - prev_grad

                    v.mul_(1 - beta2).add_(grad_diff, alpha = beta2)

                    next_n = (grad + (1 - beta2) * grad_diff) ** 2

                    n.mul_(1 - beta3).add_(next_n, alpha = beta3)

                # gradient step

                weighted_step_size = lr / (n + eps).sqrt()
                
                denom = 1 + weight_decay * lr

                data.addcmul_(weighted_step_size, (m + (1 - beta2) * v), value = -1.).div_(denom)

                # set new incremented step

                state['prev_grad'] = grad.clone()
                state['step'] += 1

        return loss
