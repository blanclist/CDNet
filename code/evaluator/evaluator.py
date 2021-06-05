import os
import torch
import numpy as np
from .mae import calc_mae
from decimal import Decimal
from .dataset import get_loader
from .smeasure import calc_smeasure
from .emeasure import calc_emeasure
from .fmeasure import calc_p_r_fmeasure

def tf(data):
    return float(data)

def tn(data):
    return np.array(data.cpu())

def td(data):
    return Decimal(data).quantize(Decimal('0.000'))

def get_n(gt, pred, n_mask):
    H, W = gt.shape
    HW = H * W
    n_gt = gt.view(1, HW).repeat(255, 1)  # [255, HW]
    n_pred = pred.view(1, HW).repeat(255, 1)  # [255, HW]
    n_pred = torch.where(n_pred <= n_mask, torch.zeros_like(n_pred), torch.ones_like(n_pred))
    return n_gt, n_pred

def evaluate_dataset(roots, dataset, batch_size, num_thread, demical, suffixes, pin):
    with torch.no_grad():
        dataloader = get_loader(roots, suffixes, batch_size, num_thread, pin=pin)
        p = np.zeros(255)
        r = np.zeros(255)
        s = 0.0
        f = np.zeros(255)
        e = np.zeros(255)
        mae = 0.0
        n_mask = torch.FloatTensor(np.array(range(255)) / 255.0).view(255, 1).repeat(1, 224 * 224).cuda()  # [255, HW]
        for batch in dataloader:
            gt, pred = batch['gt'].cuda().view(224, 224), batch['pred'].cuda().view(224, 224)

            _s = calc_smeasure(gt, pred)
            _mae = calc_mae(gt, pred)
            n_gt, n_pred = get_n(gt, pred, n_mask)
            _p, _r, _f = calc_p_r_fmeasure(n_gt, n_pred, n_mask)
            _e = calc_emeasure(n_gt, n_pred, n_mask)

            _s = tf(_s)
            _p = tn(_p)
            _r = tn(_r)
            _f = tn(_f)
            _e = tn(_e)
            _mae = tf(_mae)

            p += _p
            r += _r
            s += _s
            f += _f
            e += _e
            mae += _mae
        num = len(dataloader)
        p /= num
        r /= num
        f /= num
        e /= num
        s, mae, mean_f, max_f, mean_e, max_e = s / num, mae / num, np.mean(f), np.max(f), np.mean(e), np.max(e)
        if demical == True:
            s, mae, mean_f, max_f, mean_e, max_e = td(s), td(mae), td(mean_f), td(max_f), td(mean_e), td(max_e)
        
        results = {'s': s, 'p': p, 'r': r, 'f': f, 'e': e, 
                   'mae': mae, 
                   'mean_f': mean_f, 'max_f': max_f,
                   'mean_e': mean_e, 'max_e': max_e}
    return results