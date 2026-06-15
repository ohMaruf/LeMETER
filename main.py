from cli import parse_cli_args
from download_datasets import download_datasets
from preprocessing import preprocess_datasets
from lejepa import RUNS_DIR, pretrain_lejepa_encoder

def main():
    args = parse_cli_args()
    download_datasets()
    preprocess_datasets()
    pretrain_lejepa_encoder(
        run_name=args.name,
        config=args.config,
        resume=args.resume,
        arch=args.arch,
        dataset=args.dataset
    )


if __name__ == "__main__":
    raise SystemExit(main())