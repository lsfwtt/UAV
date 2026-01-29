import os
from PIL import Image
from argparse import ArgumentParser
from tqdm import tqdm

from label_processor import *

def parse_args():
    parser = ArgumentParser(description='transform labels from yolo type to mask')

    parser.add_argument('--image-dir', type=str, default='/home/wanmingxuan/datasets/UAV/images/train')
    parser.add_argument('--label-dir', type=str, default='/home/wanmingxuan/datasets/UAV/labels/train')
    parser.add_argument('--output-dir', type=str, default='/home/wanmingxuan/datasets/UAV/masks/train')

    return parser.parse_args()

def main():
    args = parse_args()
    image_files = os.listdir(args.image_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    tbar = tqdm(image_files, desc='Processing')
    for image_file in tbar:
        image_path = os.path.join(args.image_dir, image_file)
        label_file = os.path.splitext(image_file)[0] + '.txt'
        label_path = os.path.join(args.label_dir, label_file)
        with Image.open(image_path) as img:
            w, h = img.size
        label = LabelProcessor.get_label(label_path)
        mask = LabelProcessor.yolo_label_to_mask(label, w, h)
        mask_path = os.path.join(args.output_dir, os.path.splitext(image_file)[0] + '.png')
        mask.save(mask_path)

if __name__ == '__main__':
    main()