import os
import glob
import argparse
import shutil
import cv2


def read_yolo_labels(label_path):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            cx, cy, w, h = map(float, parts[1:5])
            boxes.append((cls, cx, cy, w, h))
    return boxes


def yolo_to_xyxy(cx, cy, bw, bh, img_w, img_h):
    x1 = int((cx - bw / 2) * img_w)
    y1 = int((cy - bh / 2) * img_h)
    x2 = int((cx + bw / 2) * img_w)
    y2 = int((cy + bh / 2) * img_h)

    x1 = max(0, min(img_w - 1, x1))
    y1 = max(0, min(img_h - 1, y1))
    x2 = max(0, min(img_w - 1, x2))
    y2 = max(0, min(img_h - 1, y2))

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    return x1, y1, x2, y2


def collect_images(images_dir):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(images_dir, e)))
    return sorted(paths)   # ✅ 按名称排序


def ensure_save_dirs(save_path):
    img_dir = os.path.join(save_path, "images")
    lbl_dir = os.path.join(save_path, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    return img_dir, lbl_dir


def visualize_yolo_dataset(image_path, label_path, check_point=None, save_path=None):
    images_dir = image_path
    labels_dir = label_path

    img_paths = collect_images(images_dir)
    if not img_paths:
        raise RuntimeError(f"No images found in {images_dir}")

    # checkpoint 定位
    start_idx = 0
    if check_point is not None:
        names = [os.path.basename(p) for p in img_paths]
        if check_point in names:
            start_idx = names.index(check_point)
        else:
            raise ValueError(f"check-point image not found: {check_point}")

    save_img_dir, save_lbl_dir, save_vis_dir = None, None, None
    if save_path:
        save_img_dir, save_lbl_dir = ensure_save_dirs(save_path)
        save_vis_dir = ensure_visualized_dir(save_path)

    idx = start_idx
    win_name = "YOLO Dataset Viewer"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    while True:
        img_path = img_paths[idx]
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(labels_dir, base + ".txt")

        img = cv2.imread(img_path)
        if img is None:
            idx = (idx + 1) % len(img_paths)
            continue

        h, w = img.shape[:2]
        boxes = read_yolo_labels(label_path)

        for cls, cx, cy, bw, bh in boxes:
            x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, bw, bh, w, h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img,
                f"cls:{cls}",
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        info = f"{idx+1}/{len(img_paths)}  {os.path.basename(img_path)}  boxes:{len(boxes)}"
        cv2.putText(
            img, info, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (255, 255, 255), 2
        )

        cv2.imshow(win_name, img)
        key = cv2.waitKey(0) & 0xFF

        if key == ord("q"):
            print(f"[EXIT] current image: {os.path.basename(img_path)}")
            break

        elif key == ord("d"):
            idx = (idx + 1) % len(img_paths)

        elif key == ord("a"):
            idx = (idx - 1 + len(img_paths)) % len(img_paths)

        elif key == ord("s"):
            if save_path is None:
                print("[WARN] save-path not provided, skip saving")
            else:
                shutil.copy2(
                    img_path,
                    os.path.join(save_img_dir, os.path.basename(img_path))
                )
                if os.path.exists(label_path):
                    shutil.copy2(
                        label_path,
                        os.path.join(save_lbl_dir, os.path.basename(label_path))
                    )
                print(f"[SAVED] {os.path.basename(img_path)}")
            idx = (idx + 1) % len(img_paths)
        
        elif key == ord("v"):
            if save_path is None or save_vis_dir is None:
                print("[WARN] save-path not provided, skip visualized saving")
            else:
                vis_name = os.path.splitext(os.path.basename(img_path))[0] + "_vis.png"
                vis_path = os.path.join(save_vis_dir, vis_name)
                cv2.imwrite(vis_path, img)
                print(f"[VIS SAVED] {vis_name}")

    cv2.destroyAllWindows()

def ensure_visualized_dir(save_path):
    vis_dir = os.path.join(save_path, "visualized")
    os.makedirs(vis_dir, exist_ok=True)
    return vis_dir


def parse_args():
    parser = argparse.ArgumentParser("YOLO Dataset Visualizer")
    parser.add_argument("--image-path", required=True, help="path to images directory")
    parser.add_argument("--label-path", required=True, help="path to labels directory")
    parser.add_argument("--check-point", default=None, help="image name to start from, e.g. 000123.jpg")
    parser.add_argument("--save-path", default="", help="save selected images/labels")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    visualize_yolo_dataset(
        image_path=args.image_path,
        label_path=args.label_path,
        check_point=args.check_point,
        save_path=args.save_path,
    )

