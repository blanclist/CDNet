import torchvision.transforms as transforms
from os import listdir
from os.path import join
from torch.utils import data
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def build_file_paths(gt_base, pred_base, gt_suffix, pred_suffix):
    gt_names_no_suffix = list(map(lambda x: x[:-len(gt_suffix)], filter(lambda x: x.endswith(gt_suffix), listdir(gt_base))))
    pred_names_no_suffix = list(map(lambda x: x[:-len(pred_suffix)], filter(lambda x: x.endswith(pred_suffix), listdir(pred_base))))
    names = list(filter(lambda x: x in pred_names_no_suffix, gt_names_no_suffix))
    gt_paths = list(map(lambda x: join(gt_base, x + gt_suffix), names))
    pred_paths = list(map(lambda x: join(pred_base, x + pred_suffix), names))
    return gt_paths, pred_paths, names

class ImageData(data.Dataset):
    def __init__(self, roots, suffixes):
        gt_base = roots['gt']
        pred_base = roots['pred']
        gt_suffix = suffixes['gt']
        pred_suffix = suffixes['pred']        
        self.gt_paths, self.pred_paths, self.names = build_file_paths(gt_base, pred_base, gt_suffix, pred_suffix)

    def __getitem__(self, item):
        gt = Image.open(self.gt_paths[item]).convert('L')
        pred = Image.open(self.pred_paths[item]).convert('L')

        transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor()
        ])
        gt, pred = transform(gt), transform(pred)

        name = self.names[item]
        return gt, pred, name

    def __len__(self):
        return len(self.names)


def get_loader(roots, suffixes, batch_size, num_thread, pin=True):
    dataset = ImageData(roots, suffixes)
    data_loader = data.DataLoader(dataset=dataset, shuffle=False, batch_size=1, num_workers=num_thread, pin_memory=pin)
    return data_loader