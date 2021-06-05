import torch

def calc_emeasure(n_gt, n_pred, n_mask):
    n_gt = n_gt - torch.mean(n_gt, dim=1, keepdims=True)  # [255, HW]
    n_pred = n_pred - torch.mean(n_pred, dim=1, keepdims=True)  # [255, HW]
    align_matrix = 2 * n_gt * n_pred / (n_gt * n_gt + n_pred * n_pred + 1e-10)  # [255, HW]
    enhanced = ((align_matrix + 1) * (align_matrix + 1)) / 4  # [255, HW]
    scores = torch.sum(enhanced, dim=1) / (enhanced.shape[1] - 1 + 1e-10)  # [255]
    return scores