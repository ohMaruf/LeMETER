import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import globals
from eval import benchmark_inference, evaluate
from hardware_acceleration import Config, enable_hardware_acceleration
from meter import GaussianMeter
from dataset import NormalizedNyuDataset

RUNS_DIR = Path("runs")

ABLATIONS = {
    "nyu_xxs_20meter60ep_derf_relu": dict(enable_derf=True, enable_gelu=False),
    "nyu_xxs_20meter60ep_derf_gelu": dict(enable_derf=True, enable_gelu=True),
    "nyu_xxs_20meter60ep_ln_relu": dict(enable_derf=False, enable_gelu=False),
    "nyu_xxs_20meter60ep_ln_gelu": dict(enable_derf=False, enable_gelu=True),
}


def main():
    torch.manual_seed(globals.SEED)
    device = enable_hardware_acceleration(Config.DEFAULT)

    test_set = NormalizedNyuDataset("test", normalization="imagenet")
    test = DataLoader(test_set, batch_size=16, shuffle=False, drop_last=True)

    results = {}
    for run_name, flags in ABLATIONS.items():
        checkpoint_path = RUNS_DIR / run_name / "best_checkpoint.pt"
        model = GaussianMeter(device, "xxs", **flags).to(device)
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        metrics = evaluate(model, test, device)
        metrics["epoch"] = ckpt["epoch"]
        metrics["best_val_rmse"] = ckpt["best_val_rmse"]

        if device.type == "cuda":
            fps_model = torch.compile(model, dynamic=False, mode="reduce-overhead")
            warmup_steps = 20
        else:
            fps_model = model
            warmup_steps = 5
        metrics["fps"] = benchmark_inference(fps_model, test_set, device, warmup_steps=warmup_steps)
        results[run_name] = metrics

        print(
            f"{run_name}: rmse={metrics['rmse']:.4f} rel={metrics['rel']:.4f} "
            f"delta1={metrics['delta1']:.4f} fps={metrics['fps']:.2f}"
        )

    out_path = Path(f"ablation_eval_results_{device.type}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
