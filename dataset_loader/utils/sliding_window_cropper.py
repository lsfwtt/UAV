import os
from PIL import Image
from argparse import ArgumentParser

class SlidingWindowCropper:
    @staticmethod
    def cropper(image_path, label_path, window_size, step):
        name = os.path.splitext(os.path.basename(image_path))[0]
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            w, h = img.size

        label = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    cls_id = int(parts[0])  # 类别
                    x_center = float(parts[1])  # x中心
                    y_center = float(parts[2])  # y中心
                    obj_w = float(parts[3])  # 宽度
                    obj_h = float(parts[4])  # 高度
                    label.append([cls_id, x_center, y_center, obj_w, obj_h])
        
        window_x_list = list(range(0, w - window_size + 1, step))
        window_y_list = list(range(0, h - window_size + 1, step))
        # 增补边界窗口
        if window_x_list[-1] != w - window_size:
            window_x_list.append(w - window_size)
        if window_y_list[-1] != h - window_size:
            window_y_list.append(h - window_size)
        
        cropped = []
        for window_x in window_x_list:
            for window_y in window_y_list:
                cropped_img = img.crop((window_x, window_y, window_x + window_size, window_y + window_size))
                cropped_label = SlidingWindowCropper.crop_label(w, h, label, window_x, window_y, window_size)
                if cropped_label is False:
                    continue
                cropped.append((name, cropped_img, cropped_label))
        return cropped

    @staticmethod
    def crop_label(img_w, img_h, label, window_x, window_y, window_size):
        cropped_labels = []
        for obj in label:
            cls_id, x_center, y_center, obj_w, obj_h = obj
            x1 = (x_center - obj_w / 2) * img_w
            y1 = (y_center - obj_h / 2) * img_h
            x2 = (x_center + obj_w / 2) * img_w
            y2 = (y_center + obj_h / 2) * img_h
            relation = SlidingWindowCropper._rect_relation(x1, y1, x2, y2, window_x, window_y, window_size)
            if relation == 'intersect but not contain':
                return False
            if relation == 'contain':
                cropped_labels.append([cls_id, 
                                       ((x1 - window_x) + (x2 - window_x)) / 2 / window_size,
                                       ((y1 - window_y) + (y2 - window_y)) / 2 / window_size,
                                       (x2 - x1) / window_size,
                                       (y2 - y1) / window_size])
        return cropped_labels

    @staticmethod
    def _rect_relation(x1, y1, x2, y2, window_x, window_y, window_size):
        if x1 >= window_x + window_size or x2 <= window_x or y1 >= window_y + window_size or y2 <= window_y:
            return 'not intersect'
        if x1 >= window_x and x2 <= window_x + window_size and y1 >= window_y and y2 <= window_y + window_size:
            return 'contain'
        return 'intersect but not contain'
    
    @staticmethod
    def save_cropped_images(cropped, output_dir):
        img_path = os.path.join(output_dir, 'images')
        label_path = os.path.join(output_dir, 'labels')
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        if not os.path.exists(label_path):
            os.makedirs(label_path)
        for i, (name, img, label) in enumerate(cropped):
            img.save(os.path.join(img_path, f'{name}_{i}.png'))
            with open(os.path.join(label_path, f'{name}_{i}.txt'), 'w') as f:
                for obj in label:
                    f.write(f"{obj[0]} {obj[1]} {obj[2]} {obj[3]} {obj[4]}\n")

def parse_args():
    parser = ArgumentParser(description='crop images and labels using sliding window')

    parser.add_argument('--image-dir', type=str, default='/home/lsfwtt/UAV/datasets/dataset_test/images/train')
    parser.add_argument('--label-dir', type=str, default='/home/lsfwtt/UAV/datasets/dataset_test/labels/train')
    parser.add_argument('--window-size', type=int, default=256)
    parser.add_argument('--step', type=int, default=128)
    parser.add_argument('--output-dir', type=str, default='/home/lsfwtt/UAV/datasets/dataset_test_cropped')

    return parser.parse_args()

def main():
    args = parse_args()
    for image_file in os.listdir(args.image_dir):
        image_path = os.path.join(args.image_dir, image_file)
        label_file = os.path.splitext(image_file)[0] + '.txt'
        label_path = os.path.join(args.label_dir, label_file)
        cropped = SlidingWindowCropper.cropper(image_path, label_path, args.window_size, args.step)
        SlidingWindowCropper.save_cropped_images(cropped, args.output_dir)

if __name__ == '__main__':
    main()