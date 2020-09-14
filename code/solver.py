import torch
import network
import numpy as np
import cv2
from dataset import get_loader
from os.path import exists
from os.path import join

class Solver(object):
    def __init__(self):
        self.net = network.Net().cuda()

    def test(self, ckpt_path, test_roots, batch_size, test_thread_num, save_base):
        with torch.no_grad():
            self.net.eval()
            ckpt = torch.load(ckpt_path)
            self.net.load_state_dict(ckpt['state_dict'])

            test_loader = get_loader(roots=test_roots, request=('img', 'depth', 'name'), batch_size=batch_size, num_thread=test_thread_num)
            
            if exists(save_base) == False:
                os.makedirs(save_base)
            
            for data_batch in test_loader:
                img, depth, names = data_batch['img'].cuda(), data_batch['depth'].cuda(), data_batch['name']
                predictions = self.net(img, depth)
                for i in range(len(predictions)):
                    batch_preds = predictions[i].permute(0, 2, 3, 1).cpu()  # [N, 1, 224, 224]
                    batch_path_list = [join(save_base, name + '_pred_{i}.png'.format(i=i)) for name in names]
                    for j in range(len(batch_path_list)):
                        pred_path = batch_path_list[j]
                        pred = batch_preds[j] * 255
                        cv2.imwrite(filename=pred_path, img=np.array(pred))