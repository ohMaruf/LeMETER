import torch
import tqdm
import logger
import globals
import json
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader
from dataset import AugmentedNyuDataset
from meter import LeMeterEncoder, Meter
from hardware_acceleration import Config, enable_hardware_acceleration
from sigreg import SigReg
from pathlib import Path

OUTPUT_DIR = Path("runs/lemeter")
CHECKPOINT_PATH = OUTPUT_DIR / "last_checkpoint.pt"


def pretrain_lejepa_encoder():
    RESUME = True
    DEVICE = enable_hardware_acceleration(Config.DEFAULT)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    meter = Meter.load(DEVICE, "nyu", "xxs")
    meter.train()

    encoder = LeMeterEncoder(DEVICE, meter.encoder).to(DEVICE)
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

    history = {
        "sigreg_loss": [],
        "lejepa_loss": [],
    }

    start_epoch = 0
    if RESUME and CHECKPOINT_PATH.exists():
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        encoder.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["optimizer_state"])
        history = ckpt.get("history", history)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        logger.warn(f"resumed from {CHECKPOINT_PATH} at epoch #{start_epoch}")

    for epoch in range(start_epoch, globals.NUM_EPOCHS):
        encoder.train()  # , probe.train()
        for views, y in tqdm.tqdm(train, total=len(train)):
            views = views.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            # _, proj = encoder(views)
            _, proj, _ = encoder(views)

            inv_loss = (proj.mean(0) - proj).square().mean()
            sigreg_loss = sigreg(proj)
            lejepa_loss = sigreg_loss * globals.LAMBDA + inv_loss * (
                1 - globals.LAMBDA
            )
            # y_rep, yhat = y.repeat_interleave(VIEWS), probe(emb.detach())
            # probe_loss = TF.cross_entropy(yhat, y_rep)
            loss = lejepa_loss  # + probe_loss

            if torch.isnan(loss):
                logger.error(f"NaN loss detected at epoch {epoch}")
                break

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            opt.step()
            scheduler.step()

            # logger.info(f'[{epoch}/{NUM_EPOCHS}] pretrain/probe {probe_loss.item()}')
            logger.info(f"[{epoch}/{globals.NUM_EPOCHS}] pretrain/lejepa {lejepa_loss.item()}")
            logger.info(f"[{epoch}/{globals.NUM_EPOCHS}] pretrain/sigreg {sigreg_loss.item()}")
            logger.info(f"[{epoch}/{globals.NUM_EPOCHS}] pretrain/inv_loss {inv_loss.item()}")

        history["sigreg_loss"].append(sigreg_loss.item())
        history["lejepa_loss"].append(lejepa_loss.item())

        checkpoint = {
            "epoch": epoch,
            "model_state": encoder.state_dict(),
            "optimizer_state": opt.state_dict(),
        }

        torch.save(checkpoint, CHECKPOINT_PATH)

    with open(OUTPUT_DIR / "losses.json", "w") as losses_file:
        json.dump(history, losses_file)


def main():
    pretrain_lejepa_encoder()

    # todo list pretraining
    # [x] add checkpoints
    # [ ] track loss across epochs to build charts loss vs epochs

    # todo list training
    # [ ] track loss across epochs to build charts loss vs epochs
    # [ ] track accuracy on test set over epochs to build chart accuracy vs epochs


if __name__ == "__main__":
    raise SystemExit(main())
