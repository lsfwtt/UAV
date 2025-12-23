from PIL import Image, ImageOps, ImageFilter
import random

def augment(self, img, mask, base_size):
    # random mirror
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    # random scale
    long_size = random.randint(int(base_size * 0.5), int(base_size * 1.5)) # 0.5   2.0
    img = img.resize((long_size, long_size), Image.BILINEAR)
    mask = mask.resize((long_size, long_size), Image.NEAREST)
    # pad
    if long_size < base_size:
        padw = padh = base_size - long_size
        img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
        mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
    # gaussian blur as in PSP
    if random.random() < 0.5: # 0.5
        img = img.filter(ImageFilter.GaussianBlur(
            radius=random.random()))
    return img, mask