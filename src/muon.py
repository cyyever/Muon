import torch
from typing import Any, Callable, Optional
from collections.abc import Iterable
import torch.distributed as dist
from torch.optim.optimizer import ParamsT


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int) -> torch.Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert (
        G.ndim >= 2
    )  # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = (
            b * A + c * A @ A
        )  # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(
    update: torch.Tensor,
    ns_steps: int = 5,
) -> torch.Tensor:
    if update.ndim == 4:  # for the case of conv filters
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    return update


class MuonWithAuxAdam(torch.optim.AdamW):
    """
    Distributed Muon variant that can be used for all parameters in the network, since it runs an
    internal AdamW for the parameters that are not compatible with Muon. The user must manually
    specify which parameters shall be optimized with Muon and which with Adam by passing in a
    list of param_groups with the `use_muon` flag set.

    The point of this class is to allow the user to have a single optimizer in their code, rather
    than having both a Muon and an Adam which each need to be stepped.

    ```
    """

    def __init__(self, params: ParamsT, *args, **kwargs) -> None:
        muon_group = [p for p in params if p.ndim >= 2]
        param_number = sum((p.numel() for p in muon_group), start=0)
        print("muon_group number", param_number)
        no_muon_group = [p for p in params if p.ndim < 2]
        param_number = sum((p.numel() for p in no_muon_group), start=0)
        print("no_muon_group number", param_number)
        super().__init__(params=no_muon_group, *args, **kwargs)
        self.muon_group = muon_group
        self.muon_state = {p: {} for p in self.muon_group}
        self.momentum = kwargs.get("momentum", 0.9)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        assert len(self.param_groups) == 1
        loss = super().step()
        assert loss is None
        momentum = self.momentum
        lr = self.param_groups[0]["lr"]
        weight_decay = self.param_groups[0]["weight_decay"]

        for p in self.muon_group:
            assert isinstance(p, torch.Tensor)

            if p.grad is None:
                # continue
                p.grad = torch.zeros_like(p)  # Force synchronization
            grad = p.grad
            if weight_decay != 0:
                grad = grad.add(p, alpha=weight_decay)

            if momentum != 0:
                buf = self.muon_state[p].get("momentum_buffer")

                if buf is None:
                    buf = torch.clone(grad).detach()
                    self.muon_state[p]["momentum_buffer"] = buf
                else:
                    buf.mul_(momentum).add_(grad)
                grad = buf

            update = muon_update(grad)
            p.add_(update.view(*p.shape), alpha=-lr)

        return loss
