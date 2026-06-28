import argparse

import kagglehub


# kaggle dataset handles
NYU_HANDLE = "soumikrakshit/nyu-depth-v2"
# ambityga/imagenet100: the standard 100-class ImageNet subset (~130k train /
# 5k val images) split across train.X1..X4 + val.X folders, plus Labels.json.
IMAGENET100_HANDLE = "ambityga/imagenet100"


def download_nyu_depth_v2() -> str:
    path = kagglehub.dataset_download(NYU_HANDLE, output_dir="datasets")
    print("nyu-depth-v2 downloaded to:", path)
    return path


def download_imagenet100() -> str:
    path = kagglehub.dataset_download(IMAGENET100_HANDLE)
    print("imagenet100 downloaded to:", path)
    return path


def download_datasets() -> None:
    download_nyu_depth_v2()
    download_imagenet100()


if __name__ == "__main__":
    raise SystemExit(download_datasets())
