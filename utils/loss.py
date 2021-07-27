import torch
from torch import nn, Tensor
from torch.nn import functional as F


class L2Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        loss = (pred - target) * mask
        loss = (loss ** 2) / 2 / pred.shape[0]
        return loss.sum()


class JointsMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()

    def forward(self, pred: Tensor, target: Tensor, target_weight: Tensor = None):
        B, J, *_ = pred.size()
        hmap_pred = pred.reshape((B, J, -1)).split(1, 1)
        hmap_gt = target.reshape((B, J, -1)).split(1, 1)

        loss = 0

        for j in range(J):
            hp = hmap_pred[j].squeeze()
            hg = hmap_gt[j].squeeze()

            if target_weight is not None:
                loss += 0.5 * self.criterion(hp.mul(target_weight[:, j]), hg.mul(target_weight[:, j]))
            else:
                loss += 0.5 * self.criterion(hp, hg)

        return loss / J

class JointsOHKMMSELoss(nn.Module):
    def __init__(self, topk=8):
        super().__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.topk = topk

    def ohkm(self, loss):
        ohkm_loss = 0.0

        for sl in loss:
            topk_idx = torch.topk(sl, self.topk, dim=0, sorted=False)[-1]
            ohkm_loss += torch.gather(sl, 0, topk_idx).sum() / self.topk
        
        ohkm_loss /= loss.shape[0]
        return ohkm_loss

    def forward(self, pred: Tensor, target: Tensor, target_weight: Tensor = None):
        B, J, *_ = pred.size()
        hmap_pred = pred.reshape((B, J, -1)).split(1, 1)
        hmap_gt = target.reshape((B, J, -1)).split(1, 1)

        loss = []

        for j in range(J):
            hp = hmap_pred[j].squeeze()
            hg = hmap_gt[j].squeeze()

            if target_weight is not None:
                mse_loss = 0.5 * self.criterion(hp.mul(target_weight[:, j]), hg.mul(target_weight[:, j]))
            else:
                mse_loss = 0.5 * self.criterion(hp, hg)
            loss.append(mse_loss.mean(dim=1).unsqueeze(dim=1))

        loss = torch.cat(loss, dim=1)

        return self.ohkm(loss)



class KLDiscretLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.criterion = nn.KLDivLoss(reduction='none')

    def forward(self, pred_x: Tensor, pred_y: Tensor, target_x: Tensor, target_y: Tensor, target_weight: Tensor = None):
        J = pred_x.shape[1]

        loss = 0.0

        for j in range(J):
            cx = self.log_softmax(pred_x[:, j].squeeze())
            cy = self.log_softmax(pred_y[:, j].squeeze())
            gx = target_x[:, j].squeeze()
            gy = target_y[:, j].squeeze()
            weight = target_weight[:, j].squeeze()
            loss += self.criterion(cx, gx).mean(dim=1).mul(weight).mean()
            loss += self.criterion(cy, gy).mean(dim=1).mul(weight).mean()

        return loss / J


class NMTNORMLoss(nn.Module):
    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)

        if label_smoothing > 0:
            self.criterion = nn.KLDivLoss(reduction='none')
        else:
            self.criterion = nn.NLLLoss(reduction='none', ignore_index=100000)
        
        self.confidence = 1.0 - label_smoothing

    def calc_loss(self, pred: Tensor, target: Tensor):
        num_tokens = pred.size(-1)

        # conduce label smoothing module
        gt = target.view(-1)

        if self.confidence < 1:
            one_hot = torch.randn(1, num_tokens, device=target.device).fill_(self.label_smoothing / (num_tokens - 1))
            one_hot = one_hot.repeat(gt.size(0), 1)
            one_hot = one_hot.scatter(1, gt.unsqueeze(1), self.confidence)
            gt = one_hot
        return self.criterion(pred, gt)

    def forward(self, pred_x: Tensor, pred_y: Tensor, target: Tensor, target_weight: Tensor):
        B, J, *_ = pred_x.size()

        loss = 0

        for j in range(J):
            cx = self.log_softmax(pred_x[:, j].squeeze())
            cy = self.log_softmax(pred_y[:, j].squeeze())
            gt = target[:, j].squeeze()
            weight = target_weight[:, j].squeeze()
            loss += self.calc_loss(cx, gt[:, 0]).mean(dim=1).mul(weight).mean()
            loss += self.calc_loss(cy, gt[:, 1]).mean(dim=1).mul(weight).mean()

        return loss / J


class NMTLoss(nn.Module):
    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)

        if label_smoothing > 0:
            self.criterion = nn.KLDivLoss(reduction='none')
        else:
            self.criterion = nn.NLLLoss(reduction='none', ignore_index=100000)
        
        self.confidence = 1.0 - label_smoothing

    def calc_loss(self, pred: Tensor, target: Tensor):
        num_tokens = pred.size(-1)

        # conduct label smoothing module
        gt = target.view(-1)

        if self.confidence < 1:
            one_hot = torch.randn(1, num_tokens, device=target.device).fill_(self.label_smoothing / (num_tokens - 1))
            one_hot = one_hot.repeat(gt.size(0), 1)
            one_hot = one_hot.scatter(1, gt.unsqueeze(1), self.confidence)
            gt = one_hot
        return self.criterion(pred, gt)

    def forward(self, pred_x: Tensor, pred_y: Tensor, target: Tensor, target_weight: Tensor):
        B, J, *_ = pred_x.size()
        loss = 0

        for j in range(J):
            cx = self.log_softmax(pred_x[:, j].squeeze())
            cy = self.log_softmax(pred_y[:, j].squeeze())
            gt = target[:, j].squeeze()
            weight = target_weight[:, j].squeeze()
            loss += self.calc_loss(cx, gt[:, 0]).sum(dim=1).mul(weight).sum()
            loss += self.calc_loss(cy, gt[:, 1]).sum(dim=1).mul(weight).sum()

        return loss / J



