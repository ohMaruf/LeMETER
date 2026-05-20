import kagglehub


def download_datasets():
    path = kagglehub.dataset_download(
        "soumikrakshit/nyu-depth-v2",
        output_dir="datasets",
    )

    print("nyu-depth-v2 downloaded to:", path)


if __name__ == '__main__':
    raise SystemExit(download_datasets())
