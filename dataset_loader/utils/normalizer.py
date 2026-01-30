import torch
import torchvision
import torch.utils.data as Data
from PIL import Image
import random
import os
from tqdm import tqdm
from argparse import ArgumentParser

class ImageDataset(Data.Dataset):
    def __init__(self, images_dir, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        self.image_files = os.listdir(images_dir)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image
    
    def __len__(self):
        return len(self.image_files)

def get_means_bias(images_dir, percentage):
    # 读取数据
    transfroms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    if not os.path.exists(images_dir):
        raise ValueError(f"Path '{images_dir}' is not a exist path")
    data = ImageDataset(images_dir, transform=transfroms)

    # 采样
    if percentage > 1.0:
        raise ValueError("Parameter 'percetage' should not be larger than 1.0")
    data_len = len(data)
    sample_len = int(data_len * percentage)
    sample_indices = random.sample(range(data_len), sample_len)
    sample_data = torch.utils.data.Subset(data, sample_indices)
    sample_loader = torch.utils.data.DataLoader(sample_data, batch_size=1, shuffle=False, num_workers=0)

    # 计算平均值和标准差
    means = torch.zeros(3)
    bias = torch.zeros(3)

    tbar = tqdm(sample_loader, desc='Calculating')
    for img in tbar:
        for d in range(3):
            means[d] += img[:, d, :, :].mean()
            bias[d] += img[:, d, :, :].std()
    
    means = means / sample_len
    bias = bias / sample_len

    return [means, bias]
    
def parse_args():
    parser = ArgumentParser(description='Calculate means and std for normalization of dataset')

    parser.add_argument('--images-dir', type=str, required=True)
    parser.add_argument('--percentage', type=float, default=1.0)

    return parser.parse_args()

def main():
    args = parse_args()
    print(get_means_bias(args.images_dir, args.percentage))

if __name__ == '__main__':
    main()