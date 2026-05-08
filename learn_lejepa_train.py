import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from lejepa.multivariate import SlicingUnivariateTest
from lejepa.univariate import ExtendedJarqueBera
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from model import Conv2dBNAct, DerfTransformer

# --- config ---
PRETRAIN_EPOCHS = 30
FINETUNE_EPOCHS = 20
BATCH_SIZE = 64
PRETRAIN_LR = 3e-4
FINETUNE_LR = 1e-3
WEIGHT_DECAY = 1e-4
IMAGE_SIZE = 64
NUM_WORKERS = 4
SIGREG_LAMBDA = 1.0
NUM_SLICES = 64
D_MODEL = 32
PROJ_DIM = 128        # projection head dim for SIGReg (discarded after pretraining)
ENCODER_LR = 3e-5    # small LR for encoder during finetuning
OUTPUT_DIR = Path("runs/lejepa")
RESUME = True


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --- model ---

class SimpleNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            Conv2dBNAct(3, 16, stride=2),
            nn.MaxPool2d(2),
            Conv2dBNAct(16, 32, stride=2),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.transformer = nn.Sequential(
            DerfTransformer(D_MODEL, 4 * D_MODEL, num_heads=8)
        )
        self.classifier = nn.Sequential(
            nn.Linear(D_MODEL, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def encode(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.transformer(x)
        return x.mean(dim=1)  # (B, D_MODEL)

    def forward(self, x):
        return self.classifier(self.encode(x))


class ProjectionPredictor(nn.Module):
    """Projects encoder output to PROJ_DIM, then predicts the other view's projection."""
    def __init__(self, d_model=D_MODEL, proj_dim=PROJ_DIM):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(d_model, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, proj_dim * 2),
            nn.GELU(),
            nn.Linear(proj_dim * 2, proj_dim),
        )

    def project(self, x):
        return self.projector(x)

    def forward(self, x):
        return self.predictor(self.project(x))


# --- data ---

class SingleViewDataset(Dataset):
    def __init__(self, hf_split, transform):
        self.ds = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        return self.transform(item["img"].convert("RGB")), item["label"]


class TwoViewDataset(Dataset):
    """Returns two independently augmented views of the same image (no label)."""
    def __init__(self, hf_split, transform):
        self.ds = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img = self.ds[idx]["img"].convert("RGB")
        return self.transform(img), self.transform(img)


def build_loaders(hf_ds):
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    strong_aug = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(IMAGE_SIZE, padding=4),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        normalize,
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])

    pin = torch.cuda.is_available()
    kw = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=pin)

    pretrain_loader = DataLoader(
        TwoViewDataset(hf_ds["train"], strong_aug),
        shuffle=True, **kw,
    )
    train_loader = DataLoader(
        SingleViewDataset(hf_ds["train"], strong_aug),
        shuffle=True, **kw,
    )
    eval_loader = DataLoader(
        SingleViewDataset(hf_ds["test"], eval_tf),
        shuffle=False, **kw,
    )
    return pretrain_loader, train_loader, eval_loader


# --- training ---

def pretrain_one_epoch(model, proj_predictor, loader, optimizer, sigreg, device):
    model.train()
    proj_predictor.train()
    total_loss = total_inv = total_sig = n = 0

    for v1, v2 in loader:
        v1 = v1.to(device, non_blocking=True)
        v2 = v2.to(device, non_blocking=True)

        z1 = model.encode(v1)
        z2 = model.encode(v2)

        # project to larger space for the SSL loss
        p1 = proj_predictor.project(z1)
        p2 = proj_predictor.project(z2)

        # invariance: predictor(proj1) -> proj2, symmetric
        inv_loss = 0.5 * (
            F.mse_loss(proj_predictor(z1), p2.detach()) +
            F.mse_loss(proj_predictor(z2), p1.detach())
        )

        # SIGReg on the projected space (PROJ_DIM >> D_MODEL, more power)
        sig_loss = sigreg(torch.cat([p1, p2], dim=0))

        loss = inv_loss + SIGREG_LAMBDA * sig_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        bs = v1.size(0)
        total_loss += loss.item() * bs
        total_inv += inv_loss.item() * bs
        total_sig += sig_loss.item() * bs
        n += bs

    return total_loss / n, total_inv / n, total_sig / n


def finetune_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = total_correct = n = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        bs = targets.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(1) == targets).sum().item()
        n += bs

    return total_loss / n, total_correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = total_correct = n = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        bs = targets.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(1) == targets).sum().item()
        n += bs

    return total_loss / n, total_correct / n


def plot_history(history, output_dir):
    for key, title in [("loss", "Loss"), ("accuracy", "Accuracy")]:
        entries = [(phase, history[phase]) for phase in history if key in (history[phase][0] if history[phase] else {})]
        if not entries:
            continue
        plt.figure(figsize=(8, 5))
        for phase, records in entries:
            if records and key in records[0]:
                plt.plot([r["epoch"] for r in records], [r[key] for r in records], label=phase)
        plt.xlabel("Epoch")
        plt.ylabel(title)
        plt.title(f"LeJEPA {title}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"{key}_curve.png", dpi=200)
        plt.close()


def main():
    device = get_device()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Using device: {device}")

    hf_ds = load_dataset("uoft-cs/cifar10")
    pretrain_loader, train_loader, eval_loader = build_loaders(hf_ds)

    model = SimpleNet(num_classes=10).to(device)
    proj_predictor = ProjectionPredictor().to(device)
    criterion = nn.CrossEntropyLoss()
    sigreg = SlicingUnivariateTest(
        univariate_test=ExtendedJarqueBera(),
        num_slices=NUM_SLICES,
        reduction="mean",
    ).to(device)

    history = {"pretrain": [], "finetune": [], "eval": []}
    best_acc = 0.0

    # --- phase 1: encoder pretraining with LeJEPA loss ---
    print("\n=== Phase 1: encoder pretraining ===")
    pretrain_optimizer = torch.optim.AdamW(
        list(model.features.parameters()) +
        list(model.transformer.parameters()) +
        list(proj_predictor.parameters()),
        lr=PRETRAIN_LR, weight_decay=WEIGHT_DECAY,
    )

    pretrain_ckpt = OUTPUT_DIR / "pretrain_checkpoint.pt"
    start_pretrain = 1
    if RESUME and pretrain_ckpt.exists():
        ckpt = torch.load(pretrain_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        proj_predictor.load_state_dict(ckpt["proj_predictor_state"])
        pretrain_optimizer.load_state_dict(ckpt["optimizer_state"])
        history["pretrain"] = ckpt.get("pretrain_history", [])
        start_pretrain = int(ckpt.get("epoch", 0)) + 1
        print(f"Resumed pretrain from epoch {start_pretrain}")

    for epoch in range(start_pretrain, PRETRAIN_EPOCHS + 1):
        t0 = time.time()
        loss, inv, sig = pretrain_one_epoch(model, proj_predictor, pretrain_loader, pretrain_optimizer, sigreg, device)
        elapsed = time.time() - t0
        history["pretrain"].append({"epoch": epoch, "loss": loss, "inv_loss": inv, "sig_loss": sig})
        print(
            f"[pretrain {epoch:03d}/{PRETRAIN_EPOCHS:03d}] "
            f"loss={loss:.4f} inv={inv:.4f} sig={sig:.4f} time={elapsed:.1f}s"
        )
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "proj_predictor_state": proj_predictor.state_dict(),
            "optimizer_state": pretrain_optimizer.state_dict(),
            "pretrain_history": history["pretrain"],
        }, pretrain_ckpt)

    # --- phase 2: fine-tune classifier + encoder at low LR ---
    print("\n=== Phase 2: classifier finetuning ===")
    finetune_optimizer = torch.optim.AdamW([
        {"params": model.classifier.parameters(), "lr": FINETUNE_LR},
        {"params": list(model.features.parameters()) + list(model.transformer.parameters()), "lr": ENCODER_LR},
    ], weight_decay=WEIGHT_DECAY)

    finetune_ckpt = OUTPUT_DIR / "last_checkpoint.pt"
    start_finetune = 1
    if RESUME and finetune_ckpt.exists():
        ckpt = torch.load(finetune_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        finetune_optimizer.load_state_dict(ckpt["optimizer_state"])
        history["finetune"] = ckpt.get("finetune_history", [])
        history["eval"] = ckpt.get("eval_history", [])
        best_acc = float(ckpt.get("best_eval_accuracy", 0.0))
        start_finetune = int(ckpt.get("epoch", 0)) + 1
        print(f"Resumed finetune from epoch {start_finetune}")

    for epoch in range(start_finetune, FINETUNE_EPOCHS + 1):
        t0 = time.time()
        ft_loss, ft_acc = finetune_one_epoch(model, train_loader, criterion, finetune_optimizer, device)
        ev_loss, ev_acc = evaluate(model, eval_loader, criterion, device)
        elapsed = time.time() - t0

        history["finetune"].append({"epoch": epoch, "loss": ft_loss, "accuracy": ft_acc})
        history["eval"].append({"epoch": epoch, "loss": ev_loss, "accuracy": ev_acc})

        print(
            f"[finetune {epoch:03d}/{FINETUNE_EPOCHS:03d}] "
            f"train_loss={ft_loss:.4f} train_acc={ft_acc:.4f} "
            f"eval_loss={ev_loss:.4f} eval_acc={ev_acc:.4f} time={elapsed:.1f}s"
        )
        plot_history(history, OUTPUT_DIR)

        is_best = ev_acc >= best_acc
        if is_best:
            best_acc = ev_acc

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": finetune_optimizer.state_dict(),
            "finetune_history": history["finetune"],
            "eval_history": history["eval"],
            "best_eval_accuracy": best_acc,
        }
        torch.save(ckpt, finetune_ckpt)
        if is_best:
            torch.save(ckpt, OUTPUT_DIR / "best_checkpoint.pt")

    print(f"\nDone. Best eval accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
