"""
SAM (Sharpness-Aware Minimization) optimizer wrapper, following the standard two-step
reference pattern from Foret et al. (2021) / the widely-used public implementation at
github.com/davda54/sam (attributed in task3/README.md). Wraps AdamW as the base optimizer,
per spec ("AdamW with the same learning rate and weight decay as ERM").

Usage per training step:
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.first_step()      # ascent to the perturbed point (rho-ball), zero_grad implied by caller
    criterion(model(x), y).backward()   # second forward/backward AT THE PERTURBED WEIGHTS
    optimizer.second_step()     # revert perturbation, then take the real AdamW step
"""
import torch


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer_cls=torch.optim.AdamW, rho=0.05, **base_kwargs):
        assert rho >= 0
        defaults = dict(rho=rho, **base_kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **base_kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=True):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)  # ascend to the perturbed point: theta + epsilon
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=True):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])  # revert to original theta
        self.base_optimizer.step()  # real update, using the gradient computed AT theta + epsilon
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2,
        )
        return norm
