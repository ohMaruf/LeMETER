from download_datasets import download_datasets
from preprocessing import preprocess_datasets
from lejepa import pretrain_lejepa_encoder

def main():
    download_datasets()
    preprocess_datasets()
    pretrain_lejepa_encoder()


if __name__ == "__main__":
    raise SystemExit(main())