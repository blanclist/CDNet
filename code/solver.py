import cv2
import torch
import network
import numpy as np
from os.path import join
from torch.optim import SGD
from dataset import get_loader
from utils import mkdir, write_doc, get_time
from loss import ce_loss, berHu_loss, boundary_loss

class Solver(object):
    def __init__(self):
        self.net = network.CDNet().cuda()

    def train(self, roots, vgg_path, init_epoch, end_epoch, learning_rate, batch_size, weight_decay, ckpt_root, doc_path, num_thread, pretrain_mode, pin):
        # 定义 SGD 优化器.
        paramters = self.net.get_pdepths.parameters() if pretrain_mode == True else self.net.parameters()
        optimizer = SGD(paramters, lr=learning_rate, momentum=0.9, weight_decay=weight_decay)

        # 加载 ".pth" 以初始化模型.
        if init_epoch == 0:
            self.net.apply(network.weights_init)
            # 从预训练的VGG16中加载.
            if pretrain_mode == True:
                self.net.vgg_rgb.vgg.load_state_dict(torch.load(vgg_path))
            else:
                self.net.vgg_d.vgg.load_state_dict(torch.load(vgg_path))
                ckpt = torch.load(join(ckpt_root, 'Weights_0.pth'))
                self.net.get_pdepths.load_state_dict(ckpt['state_dict'])
                optimizer.load_state_dict(ckpt['optimizer'])
        else:
            # 从已有的检查点文件中加载.
            load_ckpt_path = join(ckpt_root, 'Weights_0.pth') if pretrain_mode == True else join(ckpt_root, 'Weights_{}.pth').format(init_epoch)
            ckpt = torch.load(load_ckpt_path)
            self.net.load_state_dict(ckpt['state_dict'])
            optimizer.load_state_dict(ckpt['optimizer'])

        # 定义training dataloader.
        train_dataloader = get_loader(roots=roots,
                                      request=('img', 'gt', 'depth', 'filt'),
                                      shuffle=True,
                                      batch_size=batch_size,
                                      data_aug=True,
                                      num_thread=num_thread,
                                      pin=pin)
        
        # 训练.
        self.net.train()
        for epoch in range(init_epoch + 1, end_epoch):
            start_time = get_time()
            loss_sum = 0.0

            for data_batch in train_dataloader:
                self.net.zero_grad()

                # 获得一个batch的数据.
                img, filt, depth, gt = data_batch['img'], data_batch['filt'], data_batch['depth'], data_batch['gt']
                img, filt, depth, gt = img.cuda(), filt.cuda(), depth.cuda(), gt.cuda()

                if len(img) == 1:
                    # Batch Normalization 在训练时不支持 batchsize为 1, 因此直接跳过该样本的训练. 
                    continue

                # 前向传播.
                sal_preds, boundary_preds, depth_preds, _ = self.net(img, depth, pretrain_mode)
                
                if pretrain_mode == True:
                    depth_loss = berHu_loss(depth_preds, depth, filt)                
                    loss = depth_loss
                else:
                    depth_loss = berHu_loss(depth_preds, depth, filt)                
                    sal_loss = ce_loss(sal_preds, gt)
                    boundary_loss = boundary_loss(boundary_preds, gt, ksize=5)
                    loss = depth_loss + sal_loss + boundary_loss * 10.0

                loss.backward()
                optimizer.step()
                loss_sum = loss_sum + loss.detach().item()

            # 在每个epoch的训练后都保存检查点文件(".pth").
            mkdir(ckpt_root)
            if pretrain_mode == True:
                save_ckpt_path = join(ckpt_root, 'Weights_0.pth')
                paramters_state_dict = self.net.get_pdepths.state_dict()
            else:
                save_ckpt_path = join(ckpt_root, 'Weights_{}.pth'.format(epoch))
                paramters_state_dict = self.net.state_dict()
            torch.save({'optimizer': optimizer.state_dict(),
                        'state_dict': paramters_state_dict}, save_ckpt_path)

            # 近似地计算训练集的平均损失.
            loss_mean = loss_sum / len(train_dataloader)
            end_time = get_time()

            # 记录训练的信息到".txt"文档中.
            content = 'CkptIndex={}:    TrainLoss={}    LR={}    Time={}\n'.format(epoch, loss_mean, learning_rate, end_time - start_time)
            write_doc(doc_path, content)

    def test(self, roots, load_ckpt_path, pred_root, num_thread, batch_size, original_size, pin):
        with torch.no_grad():            
            # 加载指定的检查点文件(".pth").
            ckpt = torch.load(load_ckpt_path)
            self.net.load_state_dict(ckpt['state_dict'])
            self.net.eval()
            
            # 得到 test datasets 的名字.
            datasets = roots.keys()

            # 在每个 dataset 上对 CDNet 进行测试.
            for dataset in datasets:
                # 对当前 dataset 定义 test dataloader.
                test_loader = get_loader(roots=roots[dataset], 
                                         request=('img', 'depth', 'name', 'size'), 
                                         shuffle=False,
                                         data_aug=False, 
                                         num_thread=num_thread, 
                                         batch_size=batch_size, 
                                         pin=pin)
                # 为当前的dataset创建文件夹以保存之后产生的预测图.
                mkdir(pred_root)
                cur_dataset_pred_root = join(pred_root, dataset)
                mkdir(cur_dataset_pred_root)

                for data_batch in test_loader:
                    # 获得一个batch的数据.
                    img, depth, names = data_batch['img'].cuda(), data_batch['depth'].cuda(), data_batch['name']

                    # 前向传播.
                    _, _, _, preds_list = self.net(img, depth, pretrain_mode=False)

                    for i, preds in enumerate(preds_list):
                        # preds.shape: [N, 1, H, W]->[N, H, W, 1]
                        preds = preds.permute(0, 2, 3, 1).cpu()  # [N, 1, 224, 224]

                        # 制作预测图的保存路径.
                        pred_paths = [join(cur_dataset_pred_root, name + '_pred_{}.png'.format(i)) for name in names]
                        for j, pred_path in enumerate(pred_paths):
                            # 当 "original_size == True" 时, 将224*224大小的预测图缩放到原图尺寸.
                            H, W = data_batch['size'][0][j], data_batch['size'][1][j]
                            pred = cv2.resize(preds[j], (W, H)) if original_size else preds[j]
                            cv2.imwrite(filename=pred_path, img=np.array(pred * 255))