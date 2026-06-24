from cli import parse_cli_args
from download_datasets import download_datasets
from lejepa import pretrain_lejepa_encoder
from preprocessing import preprocess_datasets
from train_decoder import train_decoder


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
    if args.train_decoder:
        train_decoder(
            run_name=args.name,
            config=args.config,
            resume=args.resume,
            arch=args.arch,
            dataset=args.dataset,
            schedule=args.decoder_schedule,
            checkpoint_epoch=args.checkpoint_epoch,
        )


if __name__ == "__main__":
    raise SystemExit(main())