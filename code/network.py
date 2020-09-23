import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.nn import init
import config

# vgg choice
base = (64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M')
table = {'conv1_1': 0, 'conv1_2': 2, 'conv1_2_mp': 4,
         'conv2_1': 5, 'conv2_2': 7, 'conv2_2_mp': 9,
         'conv3_1': 10, 'conv3_2': 12, 'conv3_3': 14, 'conv3_3_mp': 16,
         'conv4_1': 17, 'conv4_2': 19, 'conv4_3': 21, 'conv4_3_mp': 23,
         'conv5_1': 24, 'conv5_2': 26, 'conv5_3': 28, 'conv5_3_mp': 30, 'final': 31}

def sub_vgg_mean(input):
    vgg_mean = torch.from_numpy(np.array([[[[103.939, 116.779, 123.68]]]])).type(torch.FloatTensor).permute(0, 3, 1, 2)
    return input - vgg_mean

def maxmin_norm(pred):
    N, C, H, W = pred.shape
    HW = H * W
    pred = pred.view(N, C, HW)
    max_value = torch.max(pred, dim=2, keepdim=True)[0]  # [N, C, 1]
    min_value = torch.min(pred, dim=2, keepdim=True)[0]  # [N, C, 1]
    norm_pred = (pred - min_value) / (max_value - min_value + 1e-12)  # [N, C, HW]
    norm_pred = norm_pred.view(N, C, H, W)
    return norm_pred

def resize(input, target_size=(224, 224), mode='bilinear'):
    if mode == 'nearest':
        return F.interpolate(input, (target_size[0], target_size[1]), mode=mode)
    else:
        return F.interpolate(input, (target_size[0], target_size[1]), mode=mode, align_corners=True)

def vgg(cfg, i=3, batch_norm=False):
    layers = []
    in_channels = i
    for idx, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return layers

def normal_init(param):
    init.normal_(param, 0, 0.01)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.normal_(m.weight, 0, 0.01)
        init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)
        
def fuse(F_Do_i, F_De_i, fusion_weight):
    F_fuse_i = F_Do_i * fusion_weight + F_De_i * (1.0 - fusion_weight)
    return F_fuse_i

################### Implementations of CDNet ###################

# VGG-16 backbone (fully connected layers are removed)
# serves as RGB encoder and Siamese depth encoder
class Vgg_Extractor(nn.Module):
    def __init__(self):
        super(Vgg_Extractor, self).__init__()
        self.vgg = nn.ModuleList(vgg(cfg=base))

    def forward(self, x, start_layer_name, end_layer_name):
        start_idx = table[start_layer_name]
        end_idx = table[end_layer_name]
        for idx in range(start_idx, end_idx):
            x = self.vgg[idx](x)
        return x


# Prediction layer: 
# a fully connected layer to compress the channel of input features to 1, 
# followed a sigmoid function to restrict results into [0, 1]
class Pred(nn.Module):
    def __init__(self, in_c):
        super(Pred, self).__init__()
        self.pred = nn.Sequential(nn.Conv2d(in_c, 1, 1), nn.Sigmoid())

    def forward(self, input):
        pred = self.pred(input)
        return pred


# Attention layer:
# two convolutional layers followed a sigmoid function
# transform ``f_d_4'' to ``W_d''
class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                                  nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.Sigmoid())

    def forward(self, feat):
        att = self.conv(feat)  # [N, 256, 28, 28]
        return att


# Feature Agrregation Module
# resort multi-scale techniques to integrate high-level(4-th and 5-th block) features
# proposed by ``A Simple Pooling-Based Design for Real-Time Salient Object Detection''(CVPR'19)
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
        self.get_pred = Pred(64)

    def forward(self, feats, size):
        feats = [resize(feat, [size, size]) for feat in feats]
        feat = torch.cat(feats, dim=1)

        res = feat
        for i in range(3):
             res = res + resize(self.convs[i](self.pools[i](feat)), [size, size])
        res = self.fuse(res)
        pred = self.get_pred(res)
        return res, pred


# U-Net decoder block 
# used in Depth estimation decoder
class Up(nn.Module):
    def __init__(self, in_c):
        super(Up, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_c, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                  nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.get_pred = Pred(64)

    def forward(self, feat, up_feat):
        N, C, H, W = feat.shape
        if up_feat is not None:
            up_feat = resize(up_feat, [H, W])
            feat = torch.cat([feat, up_feat], dim=1)
        feat = self.conv(feat)
        pred = self.get_pred(feat)
        return feat, pred


# U-Net decoder block 
# used to integrate low-level depth features(F_fuse_3, F_fuse_2 and F_fuse_1)
# for generating salient boundary maps
class Up2(nn.Module):
    def __init__(self, in_c):
        super(Up2, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_c, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                  nn.Conv2d(64, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.Sigmoid())
        self.get_pred = Pred(32)

    def forward(self, feat, up_feat):
        N, C, H, W = feat.shape
        if up_feat is not None:
            up_feat = resize(up_feat, [H, W])
            feat = torch.cat([feat, up_feat], dim=1)
        feat = self.conv(feat)
        pred = self.get_pred(feat)
        return feat, pred


# Decoder block containing a Boundary-enhanced block and a U-Net decoder block
# utilizes boundary-supervised features(f_b_3, f_b_2 and f_b_1) to enhance
# low-level RGB features(F_I_3, F_I_2, and F_I_1), and integrate them 
# to produce salient maps
class Up3(nn.Module):
    def __init__(self, in_c):
        super(Up3, self).__init__()
        self.bdry_conv = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                                  nn.Conv2d(32, 32, 3, 1, 1))
        self.conv = nn.Sequential(nn.Conv2d(in_c, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.get_pred = Pred(64)

    def forward(self, F_I_i, f_b_i, up_feat):
        N, C, H, W = F_I_i.shape
        bdry = self.bdry_conv(F_I_i * f_b_i)
        hat_F_I_i = F_I_i + bdry
        if up_feat is not None:
            up_feat = resize(up_feat, [H, W])
            feat = torch.cat([hat_F_I_i, up_feat], dim=1)
        feat = self.conv(feat)
        pred = self.get_pred(feat)
        return feat, pred
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
        a = torch.mean(a.view(N, 512, 196), dim=2).view(N, 512, 1, 1)  # [N, 512, 1, 1]
        b = torch.mean(b.view(N, 512, 196), dim=2).view(N, 512, 1, 1)  # [N, 512, 1, 1]
        feat = torch.cat([a, b], dim=1)  # [N, 1024, 1, 1]
        feat = self.fc(feat)
        return [self.att_1(feat), self.att_2(feat), self.att_3(feat), self.att_4(feat), self.att_5(feat)]


# Depth estimation decoder
# uses RGB features to estimated saliency-informative depth maps
class Pdpeth(nn.Module):
    def __init__(self):
        super(Pdpeth, self).__init__()
        self.get_pd_6 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2),
                                         nn.Conv2d(512, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                         nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                         nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.get_pdepth_6 = Pred(64)
        self.get_cmprs_pd = nn.ModuleList([nn.Conv2d(64, 64, 1), nn.Conv2d(128, 64, 1), nn.Conv2d(256, 64, 1), 
                                          nn.Conv2d(512, 64, 1), nn.Conv2d(512, 64, 1)])
        self.up_pd = nn.ModuleList([Up(128), Up(128), Up(128), Up(128), Up(128)])
        
    def forward(self, conv1_r, conv2_r, conv3_r, conv4_r, conv5_r):
        # pdepth
        pd_6 = self.get_pd_6(conv5_r)
        pd_4, pd_5 = self.get_cmprs_pd[3](conv4_r), self.get_cmprs_pd[4](conv5_r)
        pd_1, pd_2, pd_3 = self.get_cmprs_pd[0](conv1_r), self.get_cmprs_pd[1](conv2_r), self.get_cmprs_pd[2](conv3_r)
        # produce pseudo-depth
        pdepth_6 = self.get_pdepth_6(pd_6)
        f_pd_5, pdepth_5 = self.up_pd[4](pd_5, pd_6)
        f_pd_4, pdepth_4 = self.up_pd[3](pd_4, f_pd_5)
        f_pd_3, pdepth_3 = self.up_pd[2](pd_3, f_pd_4)
        f_pd_2, pdepth_2 = self.up_pd[1](pd_2, f_pd_3)
        f_pd_1, pdepth_1 = self.up_pd[0](pd_1, f_pd_2)
 
        pdepth_6 = maxmin_norm(pdepth_6)
        pdepth_5 = maxmin_norm(pdepth_5)
        pdepth_4 = maxmin_norm(pdepth_4)
        pdepth_3 = maxmin_norm(pdepth_3)
        pdepth_2 = maxmin_norm(pdepth_2)
        pdepth_1 = maxmin_norm(pdepth_1)
        return pdepth_1, pdepth_2, pdepth_3, pdepth_4, pdepth_5, pdepth_6


# CDNet
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # RGB encoder
        self.vgg_rgb = Vgg_Extractor()
        # Siamese depth encoder 
        self.vgg_d = Vgg_Extractor()
        
        # Compress channel number of F_I
        self.get_cmprs_r = nn.ModuleList([nn.Conv2d(64, 32, 1), nn.Conv2d(128, 32, 1), nn.Conv2d(256, 32, 1), 
                                          nn.Conv2d(512, 64, 1), nn.Conv2d(512, 64, 1)])
        # Compress channel number of F_De
        self.get_cmprs_dp = nn.ModuleList([nn.Conv2d(64, 32, 1), nn.Conv2d(128, 32, 1), nn.Conv2d(256, 32, 1), 
                                          nn.Conv2d(512, 64, 1), nn.Conv2d(512, 64, 1)])
        # Compress channel number of F_Do
        self.get_cmprs_d = nn.ModuleList([nn.Conv2d(64, 32, 1), nn.Conv2d(128, 32, 1), nn.Conv2d(256, 32, 1), 
                                           nn.Conv2d(512, 64, 1), nn.Conv2d(512, 64, 1)])

        # Attention layers
        self.get_att = Attention()

        # FAM
        self.pool_k, self.pool_r = FAM(), FAM()

        # U-net decoder for salient boundary predictions
        self.up = nn.ModuleList([Up3(96), Up3(96), Up3(96)])
        # Boundary-enhanced and U-net decoder for saliency map predictions
        self.up_kb = nn.ModuleList([Up2(64), Up2(64), Up2(96)])

        # Fully connected layers to produce adaptive fusion weights v_w_i
        self.fc = FC()

        # Depth estimation decoder
        self.get_pdepths = Pdpeth()

    def forward(self, img, depth):
        N, _, _, _ = img.shape
        
        # Feed Rgb-image(I) into RGB Encoder
        conv1_r = self.vgg_rgb(img, 'conv1_1', 'conv1_2_mp')
        conv2_r = self.vgg_rgb(conv1_r, 'conv1_2_mp', 'conv2_2_mp')
        conv3_r = self.vgg_rgb(conv2_r, 'conv2_2_mp', 'conv3_3_mp')
        conv4_r = self.vgg_rgb(conv3_r, 'conv3_3_mp', 'conv4_3_mp')  
        conv5_r = self.vgg_rgb(conv4_r, 'conv4_3_mp', 'conv5_3_mp')  
        F_I_1, F_I_2, F_I_3 = self.get_cmprs_r[0](conv1_r), self.get_cmprs_r[1](conv2_r), self.get_cmprs_r[2](conv3_r)
        
        # Feed Original-depth-map(Do) into Siamese Depth Encoder
        depth = maxmin_norm(depth)
        conv1_d = self.vgg_d(depth.repeat(1, 3, 1, 1), 'conv1_1', 'conv1_2_mp')
        conv2_d = self.vgg_d(conv1_d, 'conv1_2_mp', 'conv2_2_mp')
        conv3_d = self.vgg_d(conv2_d, 'conv2_2_mp', 'conv3_3_mp')
        conv4_d = self.vgg_d(conv3_d, 'conv3_3_mp', 'conv4_3_mp')  
        conv5_d = self.vgg_d(conv4_d, 'conv4_3_mp', 'conv5_3_mp')  
        F_Do_4, F_Do_5 = self.get_cmprs_d[3](conv4_d), self.get_cmprs_d[4](conv5_d)
        F_Do_1, F_Do_2, F_Do_3 = self.get_cmprs_d[0](conv1_d), self.get_cmprs_d[1](conv2_d), self.get_cmprs_d[2](conv3_d)
        
        # ---- Depth Map Estimation(Sec-III-B) ----
        # Obtain Estimated-depth-map(De)
        estimated_depth = self.get_pdepths(conv1_r, conv2_r, conv3_r, conv4_r, conv5_r)[0]
        # ------------------end--------------------

        # Feed Estimated-depth-map(De) into Siamese Depth Encoder
        conv1_dp = self.vgg_d(estimated_depth.repeat(1, 3, 1, 1), 'conv1_1', 'conv1_2_mp')
        conv2_dp = self.vgg_d(conv1_dp, 'conv1_2_mp', 'conv2_2_mp')
        conv3_dp = self.vgg_d(conv2_dp, 'conv2_2_mp', 'conv3_3_mp')
        conv4_dp = self.vgg_d(conv3_dp, 'conv3_3_mp', 'conv4_3_mp') 
        conv5_dp = self.vgg_d(conv4_dp, 'conv4_3_mp', 'conv5_3_mp')  
        F_De_4, F_De_5 = self.get_cmprs_dp[3](conv4_dp), self.get_cmprs_dp[4](conv5_dp)
        F_De_1, F_De_2, F_De_3 = self.get_cmprs_dp[0](conv1_dp), self.get_cmprs_dp[1](conv2_dp), self.get_cmprs_dp[2](conv3_dp)

        # ---- Dynamic Depth Feature Fusion(Sec-III-C) ----
        # Obtain Fusion-weights
        fusion_weights = self.fc(conv5_d, conv5_dp)
        
        # Use Fusion-weights to fuse depth features F_Do_i and F_De_i
        F_fuse_5 = fuse(F_Do_5, F_De_5, fusion_weights[4])
        F_fuse_4 = fuse(F_Do_4, F_De_4, fusion_weights[3])
        F_fuse_3 = fuse(F_Do_3, F_De_3, fusion_weights[2])
        F_fuse_2 = fuse(F_Do_2, F_De_2, fusion_weights[1])
        F_fuse_1 = fuse(F_Do_1, F_De_1, fusion_weights[0])
        # ----------------------end------------------------

        # ---- High-level Cross-modal Feature Fusion(Sec-III-D) ----
        # Generate initial saliency(P_init) with the fused high-level depth features
        f_d_4, P_init = self.pool_k([F_fuse_4, F_fuse_5], 28)
        
        # Prepare weights W_d
        W_d = self.get_att(f_d_4)

        # Enhance Down-sampled F_I_3 with W_d, then feed the result into subsequent RGB Encoder blocks(conv4 and conv5)
        conv4_r_t2 = self.vgg_rgb(F.max_pool2d(F_I_3, 2) * W_d, 'conv4_1', 'conv4_3_mp')
        conv5_r_t2 = self.vgg_rgb(conv4_r_t2, 'conv4_3_mp', 'conv5_3_mp')
        hat_F_I_4, hat_F_I_5 = self.get_cmprs_r[3](conv4_r_t2), self.get_cmprs_r[4](conv5_r_t2)

        # Produce refined saliency(P_ref) with the enhanced high-level RGB features
        F_C_h, P_ref = self.pool_r([hat_F_I_4, hat_F_I_5], 28)
        # --------------------------end-----------------------------

        # ---- Low-level Cross-modal Feature Fusion(Sec-III-D) ----
        # Extract boundary information from fused low-level depth features
        f_b_3, P_b_3 = self.up_kb[2](F_fuse_3, F_C_h)
        f_b_2, P_b_2 = self.up_kb[1](F_fuse_2, f_b_3)
        f_b_1, P_b_1 = self.up_kb[0](F_fuse_1, f_b_2)

        # Enhance low-level RGB features with the extracted result
        f_3, P_3 = self.up[2](F_I_3, f_b_3, F_C_h)
        f_2, P_2 = self.up[1](F_I_2, f_b_2, f_3)
        f_1, P_1 = self.up[0](F_I_1, f_b_1, f_2)
        # --------------------------end----------------------------

        # Return the predictions
        # All predictions = [P_1, P_2, P_3, P_b_1, P_b_2, P_b_3, P_init, P_ref, De]
        # Here we only return and save the final prediction(P_1), which is used to evaluate our CDNet
        predictions = [P_1]

        return predictions