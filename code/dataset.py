from os import listdir
from os.path import join
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils import data
import torchvision.transforms as transforms

def build_file_paths(base):
    img_paths = []
    names = []
    img_names = sorted(listdir(base))
    img_paths = [join(base, img_name) for img_name in img_names]
    names = [img_name[:-4] for img_name in img_names]
    return img_paths, names


class ImageData(data.Dataset):
    def __init__(self, roots, request, img_transform=None, depth_transform=None):
        self.need_name = True if 'name' in request else False
        self.need_depth = True if 'depth' in request else False

        img_paths, names = build_file_paths(roots['img'])
        if self.need_depth: depth_paths, _ = build_file_paths(roots['depth'])
        else: depth_paths = None
        
        self.img_paths = img_paths
        self.depth_paths = depth_paths
        self.names = names
        self.img_transform = img_transform
        self.depth_transform = depth_transform    

    def __getitem__(self, item):
        img = Image.open(self.img_paths[item]).convert('RGB')
        depth = Image.open(self.depth_paths[item]).convert('L') if self.need_depth else None
        name = self.names[item] if self.need_name else None
        
        if self.img_transform is not None:
            img = self.img_transform(img)
            
        if self.depth_transform is not None and self.need_depth:
            depth = self.depth_transform(depth)
        results = {}
        results['img'] = img
        if self.need_depth: results['depth'] = depth
        if self.need_name: results['name'] = name
        return results

    def __len__(self):
        return len(self.img_paths)


def get_loader(roots, request, batch_size, num_thread=4, pin=True):
    img_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    depth_transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor()
        ])
    dataset = ImageData(roots, request, img_transform, depth_transform)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_thread, pin_memory=pin)
    return data_loader