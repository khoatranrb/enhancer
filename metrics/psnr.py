import torch
import torch.nn as nn
import

mse = torch.nn.MSELoss()
class PSNR(nn.Module):
    def __init__(self):
        super(PSNR, self).__init__()
    def forward(self, pred, gt):
        mse_loss = mse(pred, gt)
        return 20 * math.log10(1.0 / math.sqrt(mse_loss))