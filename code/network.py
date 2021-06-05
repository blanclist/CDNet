import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.nn import init
np.set_printoptions(suppress=True, threshold=1e5)

"""
minmax_norm:
    将tensor (shape=[N, 1, H, W]) 中的值拉伸到 [0, 1] 之间. 用于对输入的深度图进行初步处理.
"""
def minmax_norm(pred):
    N, _, H, W = pred.shape
    pred = pred.view(N, -1)  # [N, HW]
    max_value = torch.max(pred, dim=1, keepdim=True)[0]  # [N, 1]
    min_value = torch.min(pred, dim=1, keepdim=True)[0]  # [N, 1]
    norm_pred = (pred - min_value) / (max_value - min_value + 1e-12)  # [N, HW]
    norm_pred = norm_pred.view(N, 1, H, W)
    return norm_pred

"""
weights_init:
    权重初始化.
"""
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.normal_(m.weight, 0, 0.01)
        init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

"""
resize:
    将tensor (shape=[N, C, H, W]) 双线性放缩到 "target_size" 大小 (默认: 224*224).
"""
def resize(input, target_size=(224, 224)):
    return F.interpolate(input, (target_size[0], target_size[1]), mode='bilinear', align_corners=True)


""""
VGG16:
    VGG16 backbone.
""" 
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        layers = []
        in_channel = 3
        vgg_out_channels = (64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M')
        for out_channel in vgg_out_channels:
            if out_channel == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channel, out_channel, 3, 1, 1)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channel = out_channel
        self.vgg = nn.ModuleList(layers)
        self.table = {'conv1_1': 0, 'conv1_2': 2, 'conv1_2_mp': 4,
                      'conv2_1': 5, 'conv2_2': 7, 'conv2_2_mp': 9,
                      'conv3_1': 10, 'conv3_2': 12, 'conv3_3': 14, 'conv3_3_mp': 16,
                      'conv4_1': 17, 'conv4_2': 19, 'conv4_3': 21, 'conv4_3_mp': 23,
                      'conv5_1': 24, 'conv5_2': 26, 'conv5_3': 28, 'conv5_3_mp': 30, 'final': 31}

    def forward(self, feats, start_layer_name, end_layer_name):
        start_idx = self.table[start_layer_name]
        end_idx = self.table[end_layer_name]
        for idx in range(start_idx, end_idx):
            feats = self.vgg[idx](feats)
        return feats


"""
Prediction:
    将输入特征的通道压缩到1维, 然后利用sigmoid函数产生预测图.
"""
class Prediction(nn.Module):
    def __init__(self, in_c):
        super(Prediction, self).__init__()
        self.pred = nn.Sequential(nn.Conv2d(in_c, 1, 1), nn.Sigmoid())

    def forward(self, input):
        pred = self.pred(input)
        return pred


"""
Attention:
    利用第四层和第五层的融合后的深度特征，产生 attention.
"""
class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                                  nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.Sigmoid())

    def forward(self, feat):
        att = self.conv(feat)  # [N, 256, 28, 28]
        return att


"""
FAM:
    Feature Aggregation Module, 用于聚合高层特征.
    来自于 "A Simple Pooling-Based Design for Real-Time Salient Object Detection (CVPR 2019)".
"""
class FAM(nn.Module):
    def __init__(self, in_c=128):
        super(FAM, self).__init__()
        pools, convs = [], []
        for i in (2, 4, 8):
            pools.append(nn.AvgPool2d(kernel_size=i, stride=i))
            convs.append(nn.Conv2d(in_c, in_c, 3, 1, 1))
        self.pools = nn.ModuleList(pools)
        self.convs = nn.ModuleList(convs)
        self.fuse = nn.Sequential(nn.BatchNorm2d(in_c), nn.ReLU(inplace=True),
                                  nn.Conv2d(in_c, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.get_pred = Prediction(64)

    def forward(self, feats, size):
        feats = [resize(feat, [size, size]) for feat in feats]
        feat = torch.cat(feats, dim=1)

        res = feat
        for i in range(3):
             res = res + resize(self.convs[i](self.pools[i](feat)), [size, size])
        res = self.fuse(res)
        pred = self.get_pred(res)
        return res, pred

"""
DB_pd:
    Decoder Block, 用于构建 depth estimation decoder.
"""
class DB_pd(nn.Module):
    def __init__(self, in_c):
        super(DB_pd, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_c, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                  nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.get_pred = Prediction(64)

    def forward(self, feat, up_feat):
        _, _, H, W = feat.shape
        if up_feat is not None:
            up_feat = resize(up_feat, [H, W])
            feat = torch.cat([feat, up_feat], dim=1)
        feat = self.conv(feat)
        pred = self.get_pred(feat)
        return feat, pred


"""
DB_b:
    Decoder Block, 利用融合后的底层深度特征产生边缘预测图.
"""
class DB_b(nn.Module):
    def __init__(self, in_c):
        super(DB_b, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_c, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                  nn.Conv2d(64, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.Sigmoid())
        self.get_pred = Prediction(32)

    def forward(self, feat, up_feat):
        _, _, H, W = feat.shape
        if up_feat is not None:
            up_feat = resize(up_feat, [H, W])
            feat = torch.cat([feat, up_feat], dim=1)
        feat = self.conv(feat)
        pred = self.get_pred(feat)
        return feat, pred

"""
DB_f:
    Decoder Block, 用于 low-level fusion 中对 RGB 和 Depth 特征进行融合.
"""
class DB_f(nn.Module):
    def __init__(self, in_c):
        super(DB_f, self).__init__()
        self.boundary_conv = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                                           nn.Conv2d(32, 32, 3, 1, 1))
        self.conv = nn.Sequential(nn.Conv2d(in_c, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                  nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.get_pred = Prediction(64)

    def forward(self, feat, bdry_att, up_feat):
        _, _, H, W = feat.shape
        bdry = self.boundary_conv(feat * bdry_att)
        feat = feat + bdry
        if up_feat is not None:
            up_feat = resize(up_feat, [H, W])
            feat = torch.cat([feat, up_feat], dim=1)
        feat = self.conv(feat)
        pred = self.get_pred(feat)
        return feat, pred


"""
FC:
    全连接层, 用于产生 dynamic depth feature fusion 中所需要的权重向量.
"""
class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        self.fc = nn.Sequential(nn.Conv2d(1024, 256, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.att_1 = nn.Sequential(nn.Conv2d(256, 32, 1), nn.BatchNorm2d(32), nn.Sigmoid())
        self.att_2 = nn.Sequential(nn.Conv2d(256, 32, 1), nn.BatchNorm2d(32), nn.Sigmoid())
        self.att_3 = nn.Sequential(nn.Conv2d(256, 32, 1), nn.BatchNorm2d(32), nn.Sigmoid())
        self.att_4 = nn.Sequential(nn.Conv2d(256, 64, 1), nn.BatchNorm2d(64), nn.Sigmoid())
        self.att_5 = nn.Sequential(nn.Conv2d(256, 64, 1), nn.BatchNorm2d(64), nn.Sigmoid())

    def forward(self, a, b):
        N, _, _, _ = a.shape
        a = torch.mean(a.view(N, 512, -1), dim=2).view(N, 512, 1, 1)  # [N, 512, 1, 1]
        b = torch.mean(b.view(N, 512, -1), dim=2).view(N, 512, 1, 1)  # [N, 512, 1, 1]
        feat = torch.cat([a, b], dim=1)  # [N, 1024, 1, 1]
        feat = self.fc(feat)
        return [self.att_1(feat), self.att_2(feat), self.att_3(feat), self.att_4(feat), self.att_5(feat)]


"""
Pdpeth_Decoder:
    Depth Estimation Decoder, 利用从 VGG16 中提取的 RGB 特征，对深度图进行预测.
"""
class Pdpeth_Decoder(nn.Module):
    def __init__(self):
        super(Pdpeth_Decoder, self).__init__()
        self.get_pd_6 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2),
                                      nn.Conv2d(512, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                      nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                      nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.get_pdepth_6 = Prediction(64)
        self.get_cmprs_pd = nn.ModuleList([nn.Conv2d(64, 64, 1), nn.Conv2d(128, 64, 1), nn.Conv2d(256, 64, 1), 
                                           nn.Conv2d(512, 64, 1), nn.Conv2d(512, 64, 1)])
        self.refine_pd = nn.ModuleList([DB_pd(128), DB_pd(128), DB_pd(128), DB_pd(128), DB_pd(128)])
        
    def forward(self, conv1_r, conv2_r, conv3_r, conv4_r, conv5_r):
        pd_6 = self.get_pd_6(conv5_r)
        pd_4, pd_5 = self.get_cmprs_pd[3](conv4_r), self.get_cmprs_pd[4](conv5_r)
        pd_1, pd_2, pd_3 = self.get_cmprs_pd[0](conv1_r), self.get_cmprs_pd[1](conv2_r), self.get_cmprs_pd[2](conv3_r)
        pdepth_6 = self.get_pdepth_6(pd_6)
        f_pd_5, pdepth_5 = self.refine_pd[4](pd_5, pd_6)
        f_pd_4, pdepth_4 = self.refine_pd[3](pd_4, f_pd_5)
        f_pd_3, pdepth_3 = self.refine_pd[2](pd_3, f_pd_4)
        f_pd_2, pdepth_2 = self.refine_pd[1](pd_2, f_pd_3)
        _, pdepth_1 = self.refine_pd[0](pd_1, f_pd_2)
 
        pdepth_6 = minmax_norm(pdepth_6)
        pdepth_5 = minmax_norm(pdepth_5)
        pdepth_4 = minmax_norm(pdepth_4)
        pdepth_3 = minmax_norm(pdepth_3)
        pdepth_2 = minmax_norm(pdepth_2)
        pdepth_1 = minmax_norm(pdepth_1)
        return pdepth_1, pdepth_2, pdepth_3, pdepth_4, pdepth_5, pdepth_6


"""
CDNet:
    整体的CDNet.
"""
class CDNet(nn.Module):
    def __init__(self):
        super(CDNet, self).__init__()
        self.vgg_rgb = VGG16()
        self.vgg_d = VGG16()
        self.get_cmprs_rgb = nn.ModuleList([nn.Conv2d(64, 32, 1), nn.Conv2d(128, 32, 1), nn.Conv2d(256, 32, 1), 
                                            nn.Conv2d(512, 64, 1), nn.Conv2d(512, 64, 1)])
        self.get_cmprs_pd = nn.ModuleList([nn.Conv2d(64, 32, 1), nn.Conv2d(128, 32, 1), nn.Conv2d(256, 32, 1), 
                                           nn.Conv2d(512, 64, 1), nn.Conv2d(512, 64, 1)])
        self.get_cmprs_d = nn.ModuleList([nn.Conv2d(64, 32, 1), nn.Conv2d(128, 32, 1), nn.Conv2d(256, 32, 1), 
                                          nn.Conv2d(512, 64, 1), nn.Conv2d(512, 64, 1)])

        self.get_pred_d_5 = Prediction(512)
        self.get_pred_pd_5 = Prediction(512)
        self.get_att = Attention()

        self.FAM_fd = FAM()
        self.FAM_rgb = FAM()
        self.refine_f = nn.ModuleList([DB_f(96), DB_f(96), DB_f(96)])
        self.refine_b = nn.ModuleList([DB_b(64), DB_b(64), DB_b(96)])

        self.fc = FC()
        self.get_pdepths = Pdpeth_Decoder()

    def forward(self, img, depth, pretrain_mode=False):
        _, _, H, _ = img.shape

        # 利用 RGB Encoder 提取 RGB 特征 
        conv1_rgb = self.vgg_rgb(img, 'conv1_1', 'conv1_2_mp')
        conv2_rgb = self.vgg_rgb(conv1_rgb, 'conv1_2_mp', 'conv2_2_mp')
        conv3_rgb = self.vgg_rgb(conv2_rgb, 'conv2_2_mp', 'conv3_3_mp')
        conv4_rgb = self.vgg_rgb(conv3_rgb, 'conv3_3_mp', 'conv4_3_mp')
        conv5_rgb = self.vgg_rgb(conv4_rgb, 'conv4_3_mp', 'conv5_3_mp')
        
        # 对提取出的 RGB 特征，利用 Depth Estimation Decoder 预测伪深度图 (pseudo depth)
        # 预测出的伪深度图为 pdepth_1/pdepth_2/pdepth_3/pdepth_4/pdepth_5/pdepth_6, 它们的尺度分别为 224/112/56/28/14/7
        # 注: 训练时会对这些伪深度图都进行监督，但送入 Depth Encoder 的只有 pdepth_1，即尺度为 224 的伪深度图
        pdepth_1, pdepth_2, pdepth_3, pdepth_4, pdepth_5, pdepth_6 = self.get_pdepths(conv1_rgb, conv2_rgb, conv3_rgb, conv4_rgb, conv5_rgb)
        depth_preds = [pdepth_1, pdepth_2, pdepth_3, pdepth_4, pdepth_5, pdepth_6]

        # pretrain_mode == True, 表明模型处于第一阶段训练模式，即产生伪深度图后就会停止
        if pretrain_mode == True:
            sal_preds = boundary_preds = preds_list = None
        # 处于第二阶段训练模式
        else:
            # 将真实深度图进行 minmax noramalization 后，在通道上重复三次，送入 Depth Encoder 提取特征
            # 其中 d_1/d_2/d_3/d_4/d_5 表示压缩后的真实深度图特征，它们的通道数分别为 32/32/32/64/64
            depth = minmax_norm(depth).repeat(1, 3, 1, 1)
            conv1_d = self.vgg_d(depth, 'conv1_1', 'conv1_2_mp')
            conv2_d = self.vgg_d(conv1_d, 'conv1_2_mp', 'conv2_2_mp')
            conv3_d = self.vgg_d(conv2_d, 'conv2_2_mp', 'conv3_3_mp')
            conv4_d = self.vgg_d(conv3_d, 'conv3_3_mp', 'conv4_3_mp')
            conv5_d = self.vgg_d(conv4_d, 'conv4_3_mp', 'conv5_3_mp')
            d_4, d_5 = self.get_cmprs_d[3](conv4_d), self.get_cmprs_d[4](conv5_d)
            d_1, d_2, d_3 = self.get_cmprs_d[0](conv1_d), self.get_cmprs_d[1](conv2_d), self.get_cmprs_d[2](conv3_d)

            # 将伪深度图在通道上重复三次，送入 Depth Encoder 提取特征 (伪深度图已经事先被 minmax normalization 过了)
            # 其中 pd_1/pd_2/pd_3/pd_4/pd_5 表示压缩后的伪深度图特征，它们的通道数分别为 32/32/32/64/64
            # 注: 提取真实/伪深度图的特征所使用的 Depth Encoder 是参数共用的，即文中提到的 'Siamese'，
            #     一方面是考虑到这样可以节省参数，不必用额外的第三个 Encoder 对伪深度图特征进行提取，
            #     但更主要的原因在于，我们从实验中发现如果使用额外的第三个 Encoder，性能反而会下降.
            conv1_pd = self.vgg_d(pdepth_1.repeat(1, 3, 1, 1), 'conv1_1', 'conv1_2_mp')
            conv2_pd = self.vgg_d(conv1_pd, 'conv1_2_mp', 'conv2_2_mp')
            conv3_pd = self.vgg_d(conv2_pd, 'conv2_2_mp', 'conv3_3_mp')
            conv4_pd = self.vgg_d(conv3_pd, 'conv3_3_mp', 'conv4_3_mp')
            conv5_pd = self.vgg_d(conv4_pd, 'conv4_3_mp', 'conv5_3_mp')
            pd_5 = self.get_cmprs_pd[4](conv5_pd)
            pd_4 = self.get_cmprs_pd[3](conv4_pd)
            pd_3 = self.get_cmprs_pd[2](conv3_pd)
            pd_2 = self.get_cmprs_pd[1](conv2_pd)
            pd_1 = self.get_cmprs_pd[0](conv1_pd)

            # Dynamic Depth Feature Fusion (Sec III.C)
            # 利用高层的真实/伪深度特征 conv5_d/conv5_pd 制作权重向量，动态控制两类深度特征的融合比例
            # 其中 fd_1/fd_2/fd_3/fd_4/fd_5 表示融合后的深度图特征，它们的通道数分别为 32/32/32/64/64
            # 注: 这部分受启发于 "Squeeze-and-Excitation Networks (CVPR 2018)" 所提出的 channel attention 的思想，
            atts = self.fc(conv5_d, conv5_pd)
            fd_5 = d_5 * atts[4] + pd_5 * (1.0 - atts[4])
            fd_4 = d_4 * atts[3] + pd_4 * (1.0 - atts[3])
            fd_3 = d_3 * atts[2] + pd_3 * (1.0 - atts[2])
            fd_2 = d_2 * atts[1] + pd_2 * (1.0 - atts[1])
            fd_1 = d_1 * atts[0] + pd_1 * (1.0 - atts[0])

            # 利用 conv5_d 和 conv5_pd 分别进行显著性预测并进行额外监督
            # 注: 尽管这里的深度监督 (deep supervision) 看上去并不起眼，但在实验中我们发现这一步非常关键.
            #     如果不进行这一步，伪深度图所带来的性能提升就会非常受限。
            #     我们同样尝试过在 conv4_d/conv4_pd 也进行类似的深度监督，但从实验结果来看，目前的选择就是最优的.
            pred_d_5 = self.get_pred_d_5(conv5_d)
            pred_pd_5 = self.get_pred_pd_5(conv5_pd)
            
            # --------- High-level Fusion (Sec III.D-1) ---------
            # 利用 FAM 对 fd_4 和 fd_5 进行融合, 并产生 28*28 的显著预测图, 进行额外监督.
            # 基于融合后的特征 fd_45, 制作一个 attention (att).
            fd_45, pred_fd_4 = self.FAM_fd([fd_4, fd_5], int(H/8))
            att = self.get_att(fd_45)
            
            # 利用 att 对 RGB stream 造成影响, 得到影响后的 RGB 特征 rgb_4/rgb_5
            conv4_rgb_t2 = self.vgg_rgb(F.max_pool2d(conv3_rgb, 2) * att, 'conv4_1', 'conv4_3_mp')
            conv5_rgb_t2 = self.vgg_rgb(conv4_rgb_t2, 'conv4_3_mp', 'conv5_3_mp')
            rgb_4, rgb_5 = self.get_cmprs_rgb[3](conv4_rgb_t2), self.get_cmprs_rgb[4](conv5_rgb_t2)
            
            # 利用 FAM 对 rgb_4 和 rgb_5 进行融合, 并产生 28*28 的显著预测图, 进行额外监督.
            # 2.0 和 1.7 是用来平衡 r_4_t2 和 r_5_t2 的超参数, 在原始论文中并未提及.
            rgb_45, pred_4 = self.FAM_rgb([rgb_4 * 2.0, rgb_5 * 1.7], int(H/8))

            # 注: 1. FAM 由 "A Simple Pooling-Based Design for Real-Time Salient Object Detection (CVPR 2019)" 提出，
            #        其借鉴了流行的多尺度技术 (例如 PPM)，在融合特征时可以提升性能. 尽管在底层 (1/2/3层) 使用 FAM 可以进一步提升性能,
            #        但我们只在高层特征融合时才使用 FAM, 原因在于在底层使用这种多尺度技术会拖慢模型速度，相较于性能提升来说得不偿失.
            #     2. 可以注意到我们在模型中多次使用了深度监督 (deep supervision) 的技术, 因为它确实可以带来性能提升.
            #        出于这个原因, 本文的所有对比实验都严格考虑了深度监督的使用，以保证实验公平性.
            # ---------------------------------------------------


            # --------- Low-level Fusion (Sec III.D-2) ---------
            # 利用底层的融合后深度特征产生边缘预测图 pred_b_1/pred_b_2/pred_b_3, 它们的尺度分别为 224/112/56
            b_3, pred_b_3 = self.refine_b[2](fd_3, rgb_45)
            b_2, pred_b_2 = self.refine_b[1](fd_2, b_3)
            b_1, pred_b_1 = self.refine_b[0](fd_1, b_2)

            # 将 rgb_45 结合 "RGB 底层特征 rgb_1/rgb_2/rgb_3" 与 "边缘特征 b_1/b_2/b_3", 
            # 产生显著预测图 pred_1/pred_2/pred_3, 它们的尺度分别为 224/112/56, 其中 pred_1 作为 CDNet 最终的预测结果.
            rgb_1, rgb_2, rgb_3 = self.get_cmprs_rgb[0](conv1_rgb), self.get_cmprs_rgb[1](conv2_rgb), self.get_cmprs_rgb[2](conv3_rgb)
            f_3, pred_3 = self.refine_f[2](rgb_3, b_3, rgb_45)
            f_2, pred_2 = self.refine_f[1](rgb_2, b_2, f_3)
            _, pred_1 = self.refine_f[0](rgb_1, b_1, f_2)
            # --------------------------------------------------

            sal_preds = [pred_d_5, pred_pd_5, pred_fd_4, pred_4, pred_3, pred_2, pred_1]
            boundary_preds = [pred_b_3, pred_b_2, pred_b_1] 
            preds_list = [pred_1, pdepth_1, pred_b_1, resize(pred_fd_4), resize(pred_4), resize(pred_d_5), resize(pred_pd_5)]

        # sal_preds: 显著预测图
        # boundary_preds: 边缘预测图
        # depth_preds: 预测出的伪深度图
        # preds_list: CDNet 在测试时输出的结果, 将会被保存至输出文件夹中
        return sal_preds, boundary_preds, depth_preds, preds_list