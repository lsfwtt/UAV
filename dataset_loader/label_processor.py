import os
from PIL import Image, ImageDraw

class LabelProcessor:
    @staticmethod
    def get_label(label_path):
        label = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    cls_id = int(parts[0])  # 类别
                    x_center = float(parts[1])  # x中心
                    y_center = float(parts[2])  # y中心
                    w = float(parts[3])  # 宽度
                    h = float(parts[4])  # 高度
                    label.append([cls_id, x_center, y_center, w, h])
        return label
    
    @staticmethod
    def yolo_label_to_mask(labels, img_w, img_h):
        # Implementation of converting YOLO labels to mask
        mask = Image.new('L', (img_w, img_h), 0)
        draw = ImageDraw.Draw(mask)
        for label in labels:
            cls_id, tx, ty, tw, th = label
            x1 = int((tx - tw / 2) * img_w)
            y1 = int((ty - th / 2) * img_h)
            x2 = int((tx + tw / 2) * img_w)
            y2 = int((ty + th / 2) * img_h)
            draw.rectangle([x1, y1, x2, y2], fill=255)
        return mask