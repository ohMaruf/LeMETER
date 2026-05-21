import torch
import torch.nn as nn
import tqdm
import logger
import globals

from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader
from dataset import build_augmented_dataset
from meter import MeterEncoder
from torch.amp import GradScaler, autocast
from hardware_acceleration import Config, enable_hardware_acceleration

DATASET_NAME = "sayakpaul/nyu_depth_v2"

DEVICE = "cpu"


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
        # phi_hat(t) = 1/B + sum_{b=1}^{n} exp( i * t * uj ), given a random direction aj
        # | phi(t) - phi(t) |^2 =
        #   = [(Re(phi_hat(t)) - phi(t)) + i Im(t)] * [(Re(phi_hat(t)) - phi(t)) - i Im(t)]
        #   =  (Re(phi_hat(t)) - phi(t))^2 + Im^2
        # B = batch size which in our case is idx -3 since x_t has shape [C, B, H, W]
        phi_hat_re = x_t.cos().mean(-3)
        phi_hat_im = x_t.sin().mean(-3)
        err = (phi_hat_re - self.phi).square() + phi_hat_im.square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


def main():
    DEVICE = enable_hardware_acceleration(Config.RX9060XT)

    encoder = MeterEncoder(DEVICE, "xxs").to(DEVICE)
    train_ds = build_augmented_dataset("train", globals.VIEWS)
    train = DataLoader(
        train_ds,
        batch_size=globals.BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=8,
    )

    # probe = nn.Sequential(nn.LayerNorm(EMBEDDING_DIM), nn.Linear(EMBEDDING_DIM, 1)).to(DEVICE)
    sigreg = SigReg().to(DEVICE)

    g1 = {
        "params": encoder.parameters(),
        "lr": globals.LEARNING_RATE,
        "weight_decay": 5e-2,
    }
    # g2 = {"params": probe.parameters(), "lr": 1e-3, "weight_decay": 1e-7}
    # opt = torch.optim.AdamW([g1, g2])
    opt = torch.optim.AdamW([g1])
    warmup_steps = len(train)
    total_steps = len(train) * globals.NUM_EPOCHS
    s1 = LinearLR(opt, start_factor=0.01, total_iters=warmup_steps)
    s2 = CosineAnnealingLR(opt, T_max=total_steps - warmup_steps, eta_min=1e-3)
    scheduler = SequentialLR(
        opt,
        schedulers=[s1, s2],
        milestones=[warmup_steps],
    )

    scaler = GradScaler()
    for epoch in range(globals.NUM_EPOCHS):
        encoder.train()  # , probe.train()
        for views, y in tqdm.tqdm(train, total=len(train)):
            with autocast("cuda", dtype=torch.float16):
                views = views.to(DEVICE, non_blocking=True)
                y = y.to(DEVICE, non_blocking=True)
                emb, proj = encoder(views)
                inv_loss = (proj.mean(0) - proj).square().mean()
                sigreg_loss = sigreg(proj)
                lejepa_loss = sigreg_loss * globals.LAMBDA + inv_loss * (
                    1 - globals.LAMBDA
                )
                # y_rep, yhat = y.repeat_interleave(VIEWS), probe(emb.detach())
                # probe_loss = TF.cross_entropy(yhat, y_rep)
                loss = lejepa_loss  # + probe_loss

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()

            # logger.info(f'[{epoch}/{NUM_EPOCHS}] train/probe {probe_loss.item()}')  # noqa: E501
            logger.info(
                f"[{epoch}/{globals.NUM_EPOCHS}] train/lejepa {lejepa_loss.item()}"
            )  # noqa: E501
            logger.info(
                f"[{epoch}/{globals.NUM_EPOCHS}] train/sigreg {sigreg_loss.item()}"
            )  # noqa: E501
            logger.info(
                f"[{epoch}/{globals.NUM_EPOCHS}] train/inv_loss {inv_loss.item()}"
            )  # noqa: E501


if __name__ == "__main__":
    raise SystemExit(main())
