import time
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from cli import parse_cli_args
from hardware_acceleration import enable_hardware_acceleration, print_device_memory_snapshot  # noqa: E501
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pathlib import Path
from model import Conv2dBNAct, DerfTransformer

matplotlib.use('Agg')

EPOCHS = 20
BATCH_SIZE = 128  # Double batch size to feed the hungry GPU architecture
LR = 3e-4
WEIGHT_DECAY = 1e-4
IMAGE_SIZE = 64
NUM_WORKERS = 4
OUTPUT_DIR = Path("runs/simple_net")


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
            DerfTransformer(32, 4 * 32, num_heads=8)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)


class HFCifarDataset(Dataset):
    def __init__(self, hf_split, transform):
        self.ds = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        return self.transform(item["img"].convert("RGB")), item["label"]


def build_loaders():
    normalize = transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )

    train_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(IMAGE_SIZE, padding=4),
        transforms.ToTensor(),
        normalize,
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])

    ds = load_dataset("uoft-cs/cifar10")
    train = HFCifarDataset(ds["train"], train_tf)
    test = HFCifarDataset(ds["test"], eval_tf)

    # 2. OPTIMIZATION: Maximize pipeline speed between CPU RAM and VRAM
    train_loader = DataLoader(
        train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )
    eval_loader = DataLoader(
        test,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )
    return train_loader, eval_loader


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = total_correct = total = 0
    for idx, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(images)
            loss = criterion(logits, targets)

        # Scaler prevents underflow errors from the float16 gradients
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if idx == 0 or idx == len(loader) - 1:
            step_label = "First Batch" if idx == 0 else "End of Epoch"
            print_device_memory_snapshot(device, step_label)

        bs = targets.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(1) == targets).sum().item()
        total += bs
    return total_loss / total, total_correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = total_correct = total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(images)
            loss = criterion(logits, targets)

        bs = targets.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(1) == targets).sum().item()
        total += bs
    return total_loss / total, total_correct / total


def plot_history(history, output_dir):
    epochs = [e["epoch"] for e in history["train"]]
    if not epochs:
        return
    for key, title in [("loss", "Loss"), ("accuracy", "Accuracy")]:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, [e[key] for e in history["train"]], label="train")
        plt.plot(epochs, [e[key] for e in history["eval"]], label="eval")
        plt.xlabel("Epoch")
        plt.ylabel(title)
        plt.title(f"SimpleNet {title}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"{key}_curve.png", dpi=200)
        plt.close()


def main():
    cli_args = parse_cli_args()
    device = enable_hardware_acceleration(cli_args.config)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_loader, eval_loader = build_loaders()

    model = SimpleNet(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    # Gradient scaler required to manage our Mixed Precision calculations
    scaler = torch.amp.GradScaler("cuda")

    history = {"train": [], "eval": []}
    best_acc = 0.0
    start_epoch = 1

    ckpt_path = OUTPUT_DIR / "last_checkpoint.pt"
    if cli_args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        history = ckpt.get("history", history)
        best_acc = float(ckpt.get("best_eval_accuracy", 0.0))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"Resumed from {ckpt_path} at epoch {start_epoch}")

    for epoch in range(start_epoch, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device
        )

        eval_loss, eval_acc = evaluate(model, eval_loader, criterion, device)
        elapsed = time.time() - t0

        history["train"].append({
            "epoch": epoch,
            "loss": train_loss,
            "accuracy": train_acc
        })

        history["eval"].append({
            "epoch": epoch,
            "loss": eval_loss,
            "accuracy": eval_acc
        })

        print(
            f"[{epoch:03d}/{EPOCHS:03d}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"eval_loss={eval_loss:.4f} eval_acc={eval_acc:.4f} time={elapsed:.1f}s"  # noqa: E501
        )

        plot_history(history, OUTPUT_DIR)

        is_best = eval_acc >= best_acc
        if is_best:
            best_acc = eval_acc

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "history": history,
            "best_eval_accuracy": best_acc,
        }
        torch.save(ckpt, OUTPUT_DIR / "last_checkpoint.pt")
        if is_best:
            torch.save(ckpt, OUTPUT_DIR / "best_checkpoint.pt")

    print(f"Done. Best eval accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    raise SystemExit(main())
