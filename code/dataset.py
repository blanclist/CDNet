import os
import torch
import random
import torchvision.transforms as transforms
from torch.utils import data
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

"""
build_file_paths:
    遍历 "base" 中的文件以构建 "img_paths", "names".
"""
def build_file_paths(base):
    img_paths = []
    names = []
    img_names = sorted(os.listdir(base))
    img_paths = [os.path.join(base, img_name) for img_name in img_names]
    names = [img_name[:-4] for img_name in img_names]
    return img_paths, names

"""
random_flip:
    以 0.5 的概率对输入数据进行随机水平翻转.
"""
def random_flip(img, gt, depth):
    datas = (img, gt, depth)
    if random.random() > 0.5:
        datas = tuple(map(lambda data:transforms.functional.hflip(data) if data is not None else data, datas))
    return datas


class ImageData(data.Dataset):
    def __init__(self, roots, request, aug_transform=None, transform=None, t_transform=None):
        self.need_gt = True if 'gt' in request else False
        self.need_name = True if 'name' in request else False
        self.need_depth = True if 'depth' in request else False
        self.need_filt = True if 'filt' in request else False
        self.need_size = True if 'size' in request else False

        img_paths, names = build_file_paths(roots['img'])
        if self.need_gt: gt_paths, _ = build_file_paths(roots['gt'])
        else: gt_paths = None
        if self.need_depth: depth_paths, _ = build_file_paths(roots['depth'])
        else: depth_paths = None
        
        if self.need_filt == True:
            # 处理 train_1529.txt, 为避免拟合低质量深度图做准备
            with open(roots['name_list'], 'r') as f:
                filt_names = f.read().splitlines()
                self.filt_names = [name[:-4] for name in filt_names]
        else:
            self.filt_names = None

        self.img_paths = img_paths
        self.gt_paths = gt_paths
        self.depth_paths = depth_paths
        self.names = names
        self.aug_transform = aug_transform
        self.transform = transform
        self.t_transform = t_transform    

    def __getitem__(self, item):
        img = Image.open(self.img_paths[item]).convert('RGB')
        W, H = img.size
        gt = Image.open(self.gt_paths[item]).convert('L') if self.need_gt else None
        depth = Image.open(self.depth_paths[item]).convert('L') if self.need_depth else None
        name = self.names[item] if self.need_name else None
        
        if self.aug_transform is not None:
            img, gt, depth = self.aug_transform(img, gt, depth)

        if self.transform is not None:
            img = self.transform(img)
            
        if self.t_transform is not None and self.need_gt:
            gt = self.t_transform(gt)

        if self.t_transform is not None and self.need_depth:
            depth = self.t_transform(depth)
        
        data_item = {}
        data_item['img'] = img
        if self.need_gt: data_item['gt'] = gt
        if self.need_depth: data_item['depth'] = depth
        if self.need_name: data_item['name'] = name
        if self.need_filt:
            data_item['filt'] = 1.0 if (self.names[item] in self.filt_names) else 0.0
        if self.need_size: data_item['size'] = (H, W)
        return data_item

    def __len__(self):
        return len(self.img_paths)


def get_loader(roots, request, batch_size, data_aug, shuffle, num_thread=4, pin=True):
    aug_transform = random_flip if data_aug else None
    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    t_transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor()
        ])
    dataset = ImageData(roots, request, aug_transform=aug_transform, transform=transform, t_transform=t_transform)
    data_loader = data.DataLoader(dataset=dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_thread, pin_memory=pin)
    return data_loader