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
in_channels = (64, 128, 256, 512, 512)
out_channels = (32, 32, 32, 64, 64)

# Subtract VGG-means from RGB images
def sub_vgg_mean(input):
    vgg_mean = torch.from_numpy(np.array([[[[103.939, 116.779, 123.68]]]])).type(torch.FloatTensor).permute(0, 3, 1, 2)
    return input - vgg_mean

# Max-min normalize depth maps(Do and De)
def maxmin_norm(pred):
    N, C, H, W = pred.shape
    HW = H * W
    pred = pred.view(N, C, HW)
    max_value = torch.max(pred, dim=2, keepdim=True)[0]  # [N, C, 1]
    min_value = torch.min(pred, dim=2, keepdim=True)[0]  # [N, C, 1]
    norm_pred = (pred - min_value) / (max_value - min_value + 1e-12)  # [N, C, HW]
    norm_pred = norm_pred.view(N, C, H, W)
    return norm_pred

# Resize tensor to specific spatial scale
def resize(input, target_size=(224, 224), mode='bilinear'):
    if mode == 'nearest':
        return F.interpolate(input, (target_size[0], target_size[1]), mode=mode)
    else:
        return F.interpolate(input, (target_size[0], target_size[1]), mode=mode, align_corners=True)

# Construct VGG encoder
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

# Extract features from VGG encoder(five blocks)
def get_features_from_VGG(vgg, input):
    F_1 = vgg(input, 'conv1_1', 'conv1_2_mp')
    F_2 = vgg(F_1, 'conv1_2_mp', 'conv2_2_mp')
    F_3 = vgg(F_2, 'conv2_2_mp', 'conv3_3_mp')
    F_4 = vgg(F_3, 'conv3_3_mp', 'conv4_3_mp')  
    F_5 = vgg(F_4, 'conv4_3_mp', 'conv5_3_mp')
    features = [F_1, F_2, F_3, F_4, F_5]
    return features

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
class U_Block(nn.Module):
    def __init__(self, in_channel, out_channel=64, activation=nn.ReLU(inplace=True)):
        super(U_Block, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channel, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                  nn.Conv2d(64, out_channel, 3, 1, 1), nn.BatchNorm2d(out_channel), activation)
        self.get_pred = Pred(out_channel)

    def forward(self, feat, up_feat):
        N, C, H, W = feat.shape
        if up_feat is not None:
            up_feat = resize(up_feat, [H, W])
            feat = torch.cat([feat, up_feat], dim=1)
        feat = self.conv(feat)
        pred = self.get_pred(feat)
        return feat, pred


# Boundary-enhanced block
# utilizes boundary-supervised features(f_b_3, f_b_2 and f_b_1) to enhance
# low-level RGB features(F_I_3, F_I_2, and F_I_1)
class Boundary_enhanced_block(nn.Module):
    def __init__(self):
        super(Boundary_enhanced_block, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                                  nn.Conv2d(32, 32, 3, 1, 1))

    def forward(self, F_I_i, f_b_i):
        N, C, H, W = F_I_i.shape
        hat_F_I_i = F_I_i + self.conv(F_I_i * f_b_i)
        return hat_F_I_i


# The module to be used for producing adaptive weights for depth feature fusion
class Weights(nn.Module):
    def __init__(self):
        super(Weights, self).__init__()
        self.fc = nn.Sequential(nn.Conv2d(1024, 256, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.fc_1 = nn.Sequential(nn.Conv2d(256, 32, 1), nn.BatchNorm2d(32), nn.Sigmoid())
        self.fc_2 = nn.Sequential(nn.Conv2d(256, 32, 1), nn.BatchNorm2d(32), nn.Sigmoid())
        self.fc_3 = nn.Sequential(nn.Conv2d(256, 32, 1), nn.BatchNorm2d(32), nn.Sigmoid())
        self.fc_4 = nn.Sequential(nn.Conv2d(256, 64, 1), nn.BatchNorm2d(64), nn.Sigmoid())
        self.fc_5 = nn.Sequential(nn.Conv2d(256, 64, 1), nn.BatchNorm2d(64), nn.Sigmoid())

    def forward(self, F_Do_5, F_De_5):
        N, _, _, _ = F_Do_5.shape
        gap_F_Do_5 = torch.mean(F_Do_5.view(N, 512, 196), dim=2).view(N, 512, 1, 1)  # [N, 512, 1, 1]
        gap_F_De_5 = torch.mean(F_De_5.view(N, 512, 196), dim=2).view(N, 512, 1, 1)  # [N, 512, 1, 1]
        v = torch.cat([gap_F_Do_5, gap_F_De_5], dim=1)  # [N, 1024, 1, 1]
        feat = self.fc(v)
        v_w_1 = self.fc_1(feat)
        v_w_2 = self.fc_2(feat)
        v_w_3 = self.fc_3(feat)
        v_w_4 = self.fc_4(feat)
        v_w_5 = self.fc_5(feat)
        return [v_w_1, v_w_2, v_w_3, v_w_4, v_w_5]


# Depth estimation decoder
# uses RGB features to estimated saliency-informative depth maps
class Depth_estimation_decoder(nn.Module):
    def __init__(self):
        super(Depth_estimation_decoder, self).__init__()
        
        # Additional layers appended on the 5-th block of RGB encoder
        self.get_feat = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2),
                                         nn.Conv2d(512, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                         nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                         nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        # 1*1 conv layers to compress ``F_I_i'' to 64 channels
        in_channels = (64, 128, 256, 512, 512)
        self.compr = nn.ModuleList([nn.Conv2d(in_channel, 64, 1) for in_channel in in_channels])

        # U-Net decoder
        self.u_decoder = nn.ModuleList([U_Block(128) for i in range(5)])
        
    def forward(self, F_I):
        # Get feat with enlarged receptive field
        feat = self.get_feat(F_I[4])

        # Estimate depth map(De)
        estimated_depth = None
        for i in range(4, -1, -1):
            feat, estimated_depth = self.u_decoder[i](self.compr[i](F_I[i]), feat)
 
        De = maxmin_norm(estimated_depth)
        return De


# Dynamically select and fuse features from original and estimated depth maps
class Dynamic_depth_feature_fusion(nn.Module):
    def __init__(self):
        super(Dynamic_depth_feature_fusion, self).__init__()
        # Fully connected layers to produce adaptive fusion weights v_w_i
        self.get_fusion_weights = Weights()
        # Compress channel number of F_De
        self.compr_De = nn.ModuleList([nn.Conv2d(in_channels[i], out_channels[i], 1) for i in range(5)])
        # Compress channel number of F_Do
        self.compr_Do = nn.ModuleList([nn.Conv2d(in_channels[i], out_channels[i], 1) for i in range(5)])
        
    def forward(self, F_Do, F_De):
        # Obtain Fusion-weights
        F_Do_5, F_De_5 = F_Do[4], F_De[4]
        v_w = self.get_fusion_weights(F_Do_5, F_De_5)
        
        # Compress F_Do_i and F_De_i and fuse them according to v_w_i
        F_Do = [self.compr_Do[i](F_Do[i]) for i in range(5)]
        F_De = [self.compr_De[i](F_De[i]) for i in range(5)]
        F_fuse = [F_Do[i] * v_w[i] + F_De[i] * (1.0 - v_w[i]) for i in range(5)]
        return F_fuse


# The proposed two-stage cross-modal feature fusion scheme
class Cross_modal_feature_fusion(nn.Module):
    def __init__(self):
        super(Cross_modal_feature_fusion, self).__init__()
        # Compress channel number of F_I
        self.compr_I = nn.ModuleList([nn.Conv2d(in_channels[i], out_channels[i], 1) for i in range(5)])
        
        self.high_level_fusion = High_level_fusion()
        self.low_level_fusion = Low_level_fusion()

    def forward(self, F_I, F_fuse, vgg_rgb):
        P_init, P_ref, F_C_h = self.high_level_fusion(F_I, F_fuse, vgg_rgb, self.compr_I)
        P_b, P = self.low_level_fusion(F_I, F_fuse, F_C_h, self.compr_I)
        return P_init, P_ref, P_b, P


# High-level cross-modal feature fusion scheme
class High_level_fusion(nn.Module):
    def __init__(self):
        super(High_level_fusion, self).__init__()
        # FAM
        self.FAM_1, self.FAM_2 = FAM(), FAM()

        # two convolutional layers followed a sigmoid function
        # transform ``f_d_4'' to ``W_d''
        self.transform = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                                       nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.Sigmoid())

    def forward(self, F_I, F_fuse, vgg_rgb, compr_I):
        # Generate initial saliency(P_init) with the fused high-level depth features
        F_fuse_4, F_fuse_5 = F_fuse[3], F_fuse[4]
        f_d_4, P_init = self.FAM_1([F_fuse_4, F_fuse_5], 28)
        # Prepare weighting tensor W_d
        W_d = self.transform(f_d_4)

        # Enhance down-sampled F_I_3 by W_d, then feed the enhanced result into subsequent RGB Encoder blocks(conv4 and conv5)
        hat_F_I_4 = vgg_rgb(F.max_pool2d(F_I[2], 2) * W_d, 'conv4_1', 'conv4_3_mp')
        hat_F_I_5 = vgg_rgb(hat_F_I_4, 'conv4_3_mp', 'conv5_3_mp')
        hat_F_I_4, hat_F_I_5 = compr_I[3](hat_F_I_4), compr_I[4](hat_F_I_5)

        # Produce refined saliency(P_ref) with the enhanced high-level RGB features
        F_C_h, P_ref = self.FAM_2([hat_F_I_4 * 2.0, hat_F_I_5 * 1.7], 28)
        return P_init, P_ref, F_C_h


# Low-level cross-modal feature fusion scheme
class Low_level_fusion(nn.Module):
    def __init__(self):
        super(Low_level_fusion, self).__init__()
        # U-net decoder for salient boundary predictions
        in_channels = [64, 64, 96]
        self.u_decoder_boundary = nn.ModuleList([U_Block(in_channel, 32, nn.Sigmoid()) for in_channel in in_channels])

        # Boundary-enhanced 
        self.boundary_enhance = nn.ModuleList([Boundary_enhanced_block() for i in range(3)])
        # and U-net decoder for saliency map predictions
        self.u_decoder_saliency = nn.ModuleList([U_Block(96) for i in range(3)])

    def forward(self, F_I, F_fuse, F_C_h, compr_I):
        # Extract boundary information from fused low-level depth features
        f_b = [None, None, None, F_C_h]
        P_b = [None, None, None]
        for i in range(2, -1, -1):
            f_b[i], P_b[i] = self.u_decoder_boundary[i](F_fuse[i], f_b[i + 1])

        # Enhance low-level RGB features with f_b_i and integrate the results to predict saliency maps
        F_I = [compr_I[i](F_I[i]) for i in range(3)]
        P = [None, None, None]
        temp_feat = F_C_h
        for i in range(2, -1, -1):
            hat_F_I_i = self.boundary_enhance[i](F_I[i], f_b[i])
            temp_feat, P[i] = self.u_decoder_saliency[i](hat_F_I_i, temp_feat)
        return P_b, P


# CDNet
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.vgg_rgb = Vgg_Extractor()  # RGB encoder
        self.vgg_d = Vgg_Extractor()  # Siamese depth encoder
        self.estimate_depth = Depth_estimation_decoder()  # Depth estimation decoder
        self.get_fused_depth_features = Dynamic_depth_feature_fusion()  # Dynamic depth feature fusion
        self.cross_modal_feature_fusion = Cross_modal_feature_fusion()  # Cross-modal feature fusion

    def forward(self, img, depth):
        # Feed Rgb-image(I) into RGB Encoder
        I = img
        F_I = get_features_from_VGG(self.vgg_rgb, I)

        # ---- Depth Map Estimation (Sec-III-B) ----
        De = self.estimate_depth(F_I)

        # Feed Original-depth-map(Do) into Siamese Depth Encoder
        Do = maxmin_norm(depth)
        F_Do = get_features_from_VGG(self.vgg_d, Do.repeat(1, 3, 1, 1))
        
        # Feed Estimated-depth-map(De) into Siamese Depth Encoder
        F_De = get_features_from_VGG(self.vgg_d, De.repeat(1, 3, 1, 1))

        # ---- Dynamic Depth Feature Fusion (Sec-III-C) ----
        F_fuse = self.get_fused_depth_features(F_Do, F_De)

        # ---- Cross-modal Feature Fusion (Sec-III-D) ----
        P_init, P_ref, P_b, P = self.cross_modal_feature_fusion(F_I, F_fuse, self.vgg_rgb)

        # Return the predictions
        # all_results = P + P_b + [P_init, P_ref, De]
        # Here we only return and save the final prediction(P[0]), which is used to evaluate our CDNet
        predictions = [P[0]]

        return predictions