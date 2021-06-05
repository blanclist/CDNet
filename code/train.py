import os
from solver import Solver

"""
训练设置(适用于 "train.py"):

vgg_path:
    预训练VGG16(".pth")的路径, 用于初始化参数来训练您自己的CDNet.

ckpt_root:
    保存检查点文件(".pth")的文件夹路径, 每个epoch训练后都会自动保存.
    第i个epoch训练完成后, 检查点文件会被保存在 "ckpt_root/Weights_{}.pth".format(i).
    * 注意: 第一阶段训练时，检查点文件被命名为 "Weights_0.pth"，并不断被覆盖；

pretrain_init_epoch:
    第一阶段训练的起始epoch.
    当 "pretrain_init_epoch == 0" 时, CDNet用预训练的VGG16的参数对 RGB Encoder 进行初始化;
    否则, CDNet加载 "ckpt_root/Weights_0.pth" 来进行初始化,

pretrain_end_epoch:
    第一阶段训练的结束epoch.
    论文中该阶段为35个epochs.

train_init_epoch:
    第二阶段训练的起始epoch.
    当 "train_init_epoch == 0" 时, CDNet用预训练的VGG16的参数对 Depth Encoder 进行初始化，并使用第一阶段的参数
    "ckpt_root/Weights_0.pth" 对相关模块进行初始化;
    否则, CDNet加载 "ckpt_root/Weights_{}.pth".format(train_init_epoch) 处的检查点文件(".pth")来进行初始化,

train_end_epoch:
    第二阶段训练的结束epoch.
    论文中该阶段为60个epochs.

device:
    用于训练的GPU编号.

train_doc_path:
    用于保存训练过程所产生信息的文件(".txt"文档)路径.

train_roots:
    一个dict, 包含训练集 RGB 图片, GTs 和 Depth 图片的文件夹路径, 其格式为:
    train_roots = {'img': 训练集的 RGB 图片的文件夹路径,
                   'gt': 训练集的 GTs 的文件夹路径,
                   'depth': 训练集的 Depth 图片的文件夹路径}
"""

vgg_path = './vgg16_feat.pth'
ckpt_root = './ckpt/'
pretrain_init_epoch = 0
pretrain_end_epoch = 35
train_init_epoch = 0
train_end_epoch = 61
device = '0'
doc_path = './training.txt'
learning_rate = 5e-3
weight_decay = 1e-4
batch_size = 10
num_thread = 4

# 下面是一个构建 "train_roots" 的例子.
train_roots = {'img': '/mnt/jwd/data/NJU+NLPR/img/',
               'gt': '/mnt/jwd/data/NJU+NLPR/gt/',
               'depth': '/mnt/jwd/data/NJU+NLPR/depth/',
               'name_list': './train_1529.txt'}  # 其中 train_1529.txt 记录了保留下来的 depth maps 的图片名
# ------------ 示例结束 ------------

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    solver = Solver()
    
    # 第一阶段训练
    solver.train(roots=train_roots, 
                 vgg_path=vgg_path, 
                 init_epoch=pretrain_init_epoch, 
                 end_epoch=pretrain_end_epoch, 
                 learning_rate=learning_rate, 
                 batch_size=batch_size, 
                 weight_decay=weight_decay, 
                 ckpt_root=ckpt_root, 
                 doc_path=doc_path, 
                 num_thread=num_thread, 
                 pretrain_mode=True, 
                 pin=False)
    
    # 第二阶段训练
    solver.train(roots=train_roots, 
                 vgg_path=vgg_path, 
                 init_epoch=train_init_epoch, 
                 end_epoch=train_end_epoch, 
                 learning_rate=learning_rate, 
                 batch_size=batch_size, 
                 weight_decay=weight_decay, 
                 ckpt_root=ckpt_root, 
                 doc_path=doc_path, 
                 num_thread=num_thread, 
                 pretrain_mode=False, 
                 pin=False)

    # 注: 经验上来看, 不通过第一阶段训练而直接进行第二阶段的联合训练, 性能会出现下降.