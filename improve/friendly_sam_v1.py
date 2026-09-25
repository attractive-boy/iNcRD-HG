"""F-SAM/SAM/ERM with exact parameter restoration and paired dropout draws.

F-SAM follows the public nblt/F-SAM code: initialize EMA with the first clean
gradient; subsequently subtract the previous EMA, then update the EMA using the
clean gradient. At sigma=1, subtracting the updated EMA differs only by a common
positive scale once the same EMA history is fixed. No exact orthogonality of the
finite EMA correction is asserted. AdamW is the common base optimizer.
"""
import math

import torch

KINDS = ('full', 'sam', 'erm')


def norm(values):
    return torch.stack([v.double().square().sum() for v in values]).sum().sqrt()


class SharpnessStep:
    def __init__(self, model, optimizer, kind, *, rho=.05, sigma=1., decay=.9, clip=5.):
        if kind not in KINDS or rho < 0 or not 0 <= decay < 1 or clip <= 0:
            raise ValueError('Invalid sharpness configuration')
        if not all(math.isfinite(v) for v in (rho, sigma, decay, clip)):
            raise ValueError('Expected finite sharpness settings')
        self.parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        if not self.parameters:
            raise ValueError('Identity deployment requires no optimizer')
        self.device = self.parameters[0][1].device
        if self.device.type == 'cuda' and self.device != torch.device('cuda:1'):
            raise ValueError('Optimization is restricted to cuda:1')
        if any(p.device != self.device for _, p in self.parameters):
            raise ValueError('All optimized parameters must share the declared device')
        self.optimizer, self.kind = optimizer, kind
        self.rho, self.sigma, self.decay, self.clip = rho, sigma, decay, clip
        self.momentum = {}
        self.gradient_evaluations = 0

    def rng(self):
        return (torch.random.get_rng_state(),
                torch.cuda.get_rng_state(self.device) if self.device.type == 'cuda' else None)

    def restore_rng(self, state):
        torch.random.set_rng_state(state[0])
        if state[1] is not None:
            torch.cuda.set_rng_state(state[1], self.device)

    def step(self, closure):
        momentum_before = dict(self.momentum)
        before = self.rng()
        self.optimizer.zero_grad(set_to_none=True)
        clean_loss = closure()
        if not torch.isfinite(clean_loss):
            raise FloatingPointError('Nonfinite clean loss')
        clean_loss.backward()
        self.gradient_evaluations += 1
        after_clean = self.rng()
        original, gradients, directions = [], [], []
        with torch.no_grad():
            for name, parameter in self.parameters:
                gradient = (parameter.grad.detach().clone() if parameter.grad is not None
                            else torch.zeros_like(parameter))
                if not torch.isfinite(gradient).all():
                    raise FloatingPointError('Nonfinite clean gradient')
                previous = self.momentum.get(name)
                direction = (gradient - self.sigma * previous if self.kind == 'full' and previous is not None
                             else gradient.clone())
                self.momentum[name] = (gradient.clone() if previous is None else
                                      self.decay * previous + (1-self.decay) * gradient)
                original.append(parameter.detach().clone())
                gradients.append(gradient); directions.append(direction)
            length = norm(directions)
            radius = 0. if self.kind == 'erm' else self.rho
            scale = radius / length.clamp_min(1e-12)
            perturbations = [d * scale.to(d.dtype) for d in directions]
            for (_, parameter), change in zip(self.parameters, perturbations):
                parameter.add_(change)
        try:
            self.optimizer.zero_grad(set_to_none=True)
            # Both passes differentiate the same realization of stochastic
            # dropout. ERM executes two identical passes to match gradient count.
            self.restore_rng(before)
            perturbed_loss = closure()
            if not torch.isfinite(perturbed_loss):
                raise FloatingPointError('Nonfinite perturbed loss')
            perturbed_loss.backward()
            self.gradient_evaluations += 1
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for _, p in self.parameters):
                raise FloatingPointError('Nonfinite perturbed gradient')
        except BaseException:
            self.momentum = momentum_before
            self.optimizer.zero_grad(set_to_none=True)
            raise
        finally:
            with torch.no_grad():
                for (_, parameter), value in zip(self.parameters, original):
                    parameter.copy_(value)
            self.restore_rng(after_clean)
        gradient_norm = torch.nn.utils.clip_grad_norm_([p for _, p in self.parameters], self.clip)
        self.optimizer.step()
        dot = torch.stack([(g.double() * d.double()).sum() for g, d in zip(gradients, directions)]).sum()
        return dict(clean_loss=float(clean_loss.detach()), perturbed_loss=float(perturbed_loss.detach()),
            perturbation_norm=float(norm(perturbations)), direction_norm=float(length),
            clean_direction_cosine=float(dot / (norm(gradients) * length).clamp_min(1e-30)),
            update_gradient_norm_before_clip=float(gradient_norm), gradient_evaluations=2)
