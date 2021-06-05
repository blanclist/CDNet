import os
from solver import Solver

"""
测试设置(适用于 "test.py"):

device:
    用于测试的GPU编号.

batch_size:
    测试时的batchsize.

pred_root:
    用于保存预测图(saliency maps)的文件夹路径.

ckpt_path:
    待测试的".pth"文件的路径.

original_size:
    当 "original_size == True" 时, CDNet 产生的预测图(224*224)会被缩放至原图尺寸后再保存.

test_roots:
    一个包含多个子dict的dict, 其中每个子dict应包含某个数据集的 RGB 图片和对应 Depth 图片的文件夹路径, 其格式为:
    test_roots = {
        数据集1的名称: {
            'img': 数据集1的 RGB 图片的文件夹路径,
            'depth': 数据集1的 Depth 图片的文件夹路径
        },
        数据集2的名称: {
            'img': 数据集2的 RGB 图片的文件夹路径,
            'depth': 数据集2的 Depth 图片的文件夹路径
        }
        .
        .
        .
    }
"""

device = '0'
batch_size = 10
pred_root = './pred/'
ckpt_path = './CDNet_vgg16.pth'
original_size = False
num_thread = 4

# 下面是一个构建 "test_roots" 的例子.
test_roots = dict()
datasets = ['NJU', 'NLPR', 'LFSD', 'DES', 'STERE', 'SSD', 'SIP', 'DUT_test']

for dataset in datasets:
    roots = {'img': '/mnt/jwd/data/{}/img_bilinear_224/'.format(dataset),
             'depth': '/mnt/jwd/data/{}/depth_bilinear_224/'.format(dataset)}
    test_roots[dataset] = roots
# ------------ 示例结束 ------------

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    solver = Solver()
    solver.test(roots=test_roots,
                load_ckpt_path=ckpt_path,
                pred_root=pred_root, 
                num_thread=num_thread, 
                batch_size=batch_size, 
                original_size=original_size,
                pin=False)