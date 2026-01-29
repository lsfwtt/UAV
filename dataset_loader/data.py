import torch
import torch.utils.data as Data
import torchvision.transforms as transforms
from PIL import Image
import os

from dataset_loader.data_augmentation import *
from dataset_loader.utils.label_processor import *

class Segmentation_Dataset_train(Data.Dataset):
    def __init__(self, args, mode):
        dataset_dir = args.dataset_dir
        self.base_size = args.base_size
        self.epoch_size = args.epoch_size
        self.mode = mode

        self.imgs_dir = os.path.join(dataset_dir, 'images', self.mode)
        self.labels_dir = os.path.join(dataset_dir, 'labels', self.mode)
        self.masks_dir = os.path.join(dataset_dir, 'masks', self.mode)

        self.samples = []
        for img_name in os.listdir(self.imgs_dir):
            name = os.path.splitext(img_name)[0]
            label_name = name + '.txt'
            mask_name = name + '.png'
            self.samples.append({
                'image_path': os.path.join(self.imgs_dir, img_name),
                'label_path': os.path.join(self.labels_dir, label_name),
                'mask_path': os.path.join(self.masks_dir, mask_name),
            })

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

    def __getitem__(self, i):
        index = torch.randint(0, len(self.samples), (1,)).item()
        sample = self.samples[index]
        img = Image.open(sample['image_path']).convert('RGB')
        # label = LabelProcessor.get_label(sample['label_path'])
        mask = Image.open(sample['mask_path']).convert('L')
        img, mask = self.transform(img), transforms.ToTensor()(mask)
        return img, mask

    def __len__(self):
        return self.epoch_size

class Segmentation_Dataset_val(Data.Dataset):
    def __init__(self, args, mode):
        dataset_dir = args.dataset_dir
        self.base_size = args.base_size
        self.mode = mode

        self.imgs_dir = os.path.join(dataset_dir, 'images', self.mode)
        self.labels_dir = os.path.join(dataset_dir, 'labels', self.mode)
        self.mask_dir = os.path.join(dataset_dir, 'masks', self.mode)

        self.samples = []
        for img_name in os.listdir(self.imgs_dir):
            name = os.path.splitext(img_name)[0]
            label_name = name + '.txt'
            mask_name = name + '.png'
            self.samples.append({
                'image_path': os.path.join(self.imgs_dir, img_name),
                'label_path': os.path.join(self.labels_dir, label_name),
                'mask_path': os.path.join(self.mask_dir, mask_name)
            })

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

    def __getitem__(self, i):
        sample = self.samples[i]
        img = Image.open(sample['image_path']).convert('RGB')
        # label = LabelProcessor.get_label(sample['label_path'])
        mask = Image.open(sample['mask_path']).convert('L')
        img, mask = self.transform(img), transforms.ToTensor()(mask)
        return img, mask

    def __len__(self):
        return len(self.samples)