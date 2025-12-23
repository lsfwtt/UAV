import os
import shutil
import random
from pathlib import Path


def split_yolo_dataset(
    dataset_root,
    train_ratio=0.7,
    val_ratio=0.2,
    test_ratio=0.1,
    seed=42,
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "比例之和必须为 1"

    dataset_root = Path(dataset_root)
    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"

    assert images_dir.exists(), f"{images_dir} 不存在"
    assert labels_dir.exists(), f"{labels_dir} 不存在"

    # 支持的图片后缀
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    # 收集图片（以 image 为准）
    image_files = sorted([
        p for p in images_dir.iterdir()
        if p.suffix.lower() in img_exts
    ])

    if not image_files:
        raise RuntimeError("images 目录下未找到图片")

    random.seed(seed)
    random.shuffle(image_files)

    n = len(image_files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train + n_val]
    test_files = image_files[n_train + n_val:]

    split_map = {
        "train": train_files,
        "val": val_files,
        "test": test_files,
    }

    # 创建目标目录
    for split in split_map:
        (images_dir / split).mkdir(parents=True, exist_ok=True)
        (labels_dir / split).mkdir(parents=True, exist_ok=True)

    # 移动文件
    for split, files in split_map.items():
        for img_path in files:
            label_path = labels_dir / (img_path.stem + ".txt")

            # move image
            shutil.move(
                str(img_path),
                str(images_dir / split / img_path.name)
            )

            # move label（若存在）
            if label_path.exists():
                shutil.move(
                    str(label_path),
                    str(labels_dir / split / label_path.name)
                )

    print("数据集划分完成：")
    print(f"  train: {len(train_files)}")
    print(f"  val  : {len(val_files)}")
    print(f"  test : {len(test_files)}")


if __name__ == "__main__":
    split_yolo_dataset(
        dataset_root="/home/lsfwtt/UAV/datasets/dataset_test",
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        seed=42,
    )
    