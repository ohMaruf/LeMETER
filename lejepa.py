import torch
import tqdm
import logger
import globals

from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader
from dataset import AugmentedNyuDataset
from meter import MeterEncoder
from torch.amp import GradScaler, autocast
from hardware_acceleration import Config, enable_hardware_acceleration
from sigreg import SigReg


def pretrain_lejepa_encoder():
    DEVICE = enable_hardware_acceleration(Config.RX9060XT)

    encoder = MeterEncoder(DEVICE, "xxs").to(DEVICE)
    train_ds = AugmentedNyuDataset("train", globals.VIEWS)
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
                lejepa_loss = sigreg_loss * globals.LAMBDA + inv_loss * (1 - globals.LAMBDA)
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
    raise SystemExit(pretrain_lejepa_encoder())
