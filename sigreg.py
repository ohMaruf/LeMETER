import torch
import torch.nn as nn
import globals

class SigReg(nn.Module):
    def __init__(self, knots=17):
        super().__init__()

        t = torch.linspace(0, 3, knots, dtype=globals.FLOATING_PRECISION)
        dt = 3 / (knots - 1)

        # [dt, 2 * dt, ..., 2 * dt, dt]
        weights = torch.full(
            (knots,),
            2 * dt,
            dtype=globals.FLOATING_PRECISION,
        )
        weights[[0, -1]] = dt

        # gaussian distribution characteristic function
        # phi(t) = exp(-t^2 / 2)
        window = torch.exp(-t.square() / 2)

        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        # number of random directions
        num_slices = globals.NUM_SLICES

        # matrix of unit vectors pointing towards random directions
        A = torch.randn(proj.size(-1), num_slices, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))

        # A = [a1, ..., aM]
        # z @ a = projection of z on a
        # z @ A = [u1, ..., uM] = x_t
        # x_t = [C, B, H, W]
        x_t = (proj @ A).unsqueeze(-1) * self.t

        # phi(t) = exp(-t^2 / 2)
        # phi_hat(t) := empirical characteristic function
        # phi_hat(t) = 1/B * sum_{b=1}^{B} exp( i * t * uj ), given a random direction aj
        # with Euler complex exponential formula: exp( i * x ) = cos(x) + i sin(x)
        # phi_hat(t) = 1/B * sum_{b=1}^{B} cos( t * uj ) + i * 1/B * sum_{b=1}^{B} sin( t * uj ),
        # let phi_hat(t) = Re + i Im

        # | phi_hat(t) - phi(t) |^2 =
        # with |z| := z * z* (z* is z conjugated)
        #   = [(Re - phi(t)) + i Im] * [(Re) - phi(t)) - i Im(t)]
        #   =  (Re) - phi(t))^2 + Im^2
        # B = batch size which in our case is idx -3 since x_t has shape [C, B, H, W]
        phi_hat_re = x_t.cos().mean(-3)
        phi_hat_im = x_t.sin().mean(-3)
        err = (phi_hat_re - self.phi).square() + phi_hat_im.square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()

