import torch
import torch.nn as nn
import torch.nn.functional as TF
import tqdm
import logger

from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader
from dataset import build_dataset
from meter import MeterEncoder
from globals import NUM_SLICES, FLOATING_PRECISION, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, LAMBDA, VIEWS, EMBEDDING_DIM
from torch.amp import GradScaler, autocast
from hardware_acceleration import Config, enable_hardware_acceleration

DATASET_NAME = 'sayakpaul/nyu_depth_v2'

DEVICE = 'cpu'


class SigReg(nn.Module):
    def __init__(self, knots=17):
        super().__init__()

        t = torch.linspace(0, 3, knots, dtype=FLOATING_PRECISION)
        dt = 3 / (knots - 1)

        weights = torch.full((knots,), 2 * dt, dtype=FLOATING_PRECISION)
        weights[[0, -1]] = dt

        sigma_squared = 2.0
        window = torch.exp(-t.square() / sigma_squared)

        self.register_buffer('t', t)
        self.register_buffer('phi', window)
        self.register_buffer('weights', weights * window)

    def forward(self, proj):
        num_slices = NUM_SLICES
        A = torch.randn(proj.size(-1), num_slices, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()  # noqa: E501
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


def main():
    DEVICE = enable_hardware_acceleration(Config.RX9060XT)

    encoder = MeterEncoder(DEVICE, 'xxs').to(DEVICE)
    train_ds = build_dataset('train', VIEWS)
    train = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=8
    )

    # probe = nn.Sequential(nn.LayerNorm(EMBEDDING_DIM), nn.Linear(EMBEDDING_DIM, 1)).to(DEVICE)
    sigreg = SigReg().to(DEVICE)

    g1 = {"params": encoder.parameters(), "lr": LEARNING_RATE, "weight_decay": 5e-2}
    # g2 = {"params": probe.parameters(), "lr": 1e-3, "weight_decay": 1e-7}
    # opt = torch.optim.AdamW([g1, g2])
    opt = torch.optim.AdamW([g1])
    warmup_steps = len(train)
    total_steps = len(train) * NUM_EPOCHS
    s1 = LinearLR(opt, start_factor=0.01, total_iters=warmup_steps)
    s2 = CosineAnnealingLR(opt, T_max=total_steps - warmup_steps, eta_min=1e-3)
    scheduler = SequentialLR(opt, schedulers=[s1, s2], milestones=[warmup_steps])

    scaler = GradScaler()
    for epoch in range(NUM_EPOCHS):
        encoder.train()  # , probe.train()
        for vs, y in tqdm.tqdm(train, total=len(train)):
            with autocast('cuda', dtype=torch.float16):
                vs = vs.to(DEVICE, non_blocking=True)
                y = y.to(DEVICE, non_blocking=True)
                emb, proj = encoder(vs)
                inv_loss = (proj.mean(0) - proj).square().mean()
                sigreg_loss = sigreg(proj)
                lejepa_loss = sigreg_loss * LAMBDA + inv_loss * (1 - LAMBDA)
                # y_rep, yhat = y.repeat_interleave(VIEWS), probe(emb.detach())
                # probe_loss = TF.cross_entropy(yhat, y_rep)
                loss = lejepa_loss   # + probe_loss

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()

            # logger.info(f'[{epoch}/{NUM_EPOCHS}] train/probe {probe_loss.item()}')  # noqa: E501
            logger.info(f'[{epoch}/{NUM_EPOCHS}] train/lejepa {lejepa_loss.item()}')  # noqa: E501
            logger.info(f'[{epoch}/{NUM_EPOCHS}] train/sigreg {sigreg_loss.item()}')  # noqa: E501
            logger.info(f'[{epoch}/{NUM_EPOCHS}] train/inv_loss {inv_loss.item()}')  # noqa: E501


if __name__ == '__main__':
    raise SystemExit(main())
