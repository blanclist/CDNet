import cv2
import torch
import torch.nn.functional as F
from torch import nn

"""
boundary_loss:
    边界损失，用于训练对显著性目标边界的预测.
"""
def boundary_loss(pred_list, gt, ksize):
    N, _, H, W = gt.shape
    HW = H * W

    # 制作边缘 GT
    padding = int((ksize - 1) / 2)
    gt = torch.abs(gt - F.avg_pool2d(gt, ksize, 1, padding))  
    gt = torch.where(gt > torch.zeros_like(gt), torch.ones_like(gt), torch.zeros_like(gt))  # [N, 1, H, W]

    loss_acc = 0
    for pred in pred_list:
        resized_pred = F.interpolate(pred, (H, W), mode='bilinear', align_corners=True)        
        loss = (resized_pred - gt) ** 2

        # pos_rate 和 neg_rate 用于平衡边缘/非边缘像素之间的损失
        pos_rate = torch.mean(gt.view(N, HW), dim=1).view(N, 1, 1, 1)  # [N, 1, 1, 1]
        neg_rate = 1.0 - pos_rate
        
        weight = gt * neg_rate + (1.0 - gt) * pos_rate
        loss = loss * weight
        loss_acc += loss.mean()
    return loss_acc


"""
ce_loss:
    交叉熵损失，用于训练对显著图的预测.
"""
def ce_loss(pred_list, gt):
    _, _, H, W = gt.shape
    loss_acc = 0
    for pred in pred_list:
        resized_pred = F.interpolate(pred, (H, W), mode='bilinear', align_corners=True)        
        
        # torch.clamp() 用于防止 log 溢出
        loss = - gt * torch.log(torch.clamp(resized_pred, 1e-10, 1)) - (1 - gt) * torch.log(torch.clamp(1 - resized_pred, 1e-10, 1))
        
        loss_acc += loss.mean()
    return loss_acc


"""
berHu_loss:
    berHu 损失，用于训练对深度图的预测.
    filt 为只对过滤后的深度图进行拟合; 当 filt=None 时，对所有深度图都进行拟合。
"""
def berHu_loss(pred_list, gt, filt=None):
    N, _, H, W = gt.shape
    HW = H * W
    loss_acc = 0.0
    for pred in pred_list:
        _, _, pH, pW = pred.shape
        pHW = pH * pW
        pred = F.interpolate(pred, (H, W), mode='bilinear', align_corners=True)  # [N, 1, H, W]
        l1_error = torch.abs(pred - gt)  # [N, 1, H, W]
        theta = 0.2 * torch.max(l1_error.view(N, 1, HW).detach(), dim=2)[0].view(N, 1, 1, 1)  # [N, 1, 1, 1]      
        mask_1 = torch.where(l1_error <= torch.ones_like(l1_error) * theta, torch.ones_like(l1_error), torch.zeros_like(l1_error))
        mask_2 = 1.0 - mask_1
        loss = l1_error * mask_1 + (((pred - gt) ** 2 + theta ** 2) / (2 * theta)) * mask_2  # [N, 1, H, W]
        if filt is not None:
            filt = filt.view(N, 1, 1, 1)
            loss = loss * filt.float()
        loss = loss.mean() * pHW / HW
        loss_acc += loss
    return loss_acc