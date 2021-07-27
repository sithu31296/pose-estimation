import torch 
from torch import nn, Tensor
from typing import Tuple

class Conv(nn.Sequential):
    def __init__(self, c1, c2, k=3, s=1, p=1, d=1):
        super().__init__(
            nn.Conv2d(c1, c2, k, s, p, d),
            nn.ReLU(True)
        )

class ConvBN(nn.Sequential):
    def __init__(self, c1, c2, k=3, s=1, p=1, d=1, bias=True):
        super().__init__(
            nn.Conv2d(c1, c2, k, s, p, d, bias=bias),
            nn.BatchNorm2d(c2),
            nn.ReLU(True)
        )

class DWConv(nn.Sequential):
    def __init__(self, c1, c2, k=3, s=1, p=1, d=1):
        super().__init__(
            nn.Conv2d(c1, c1, k, s, p, d, c1, bias=False),
            nn.ELU(True),
            nn.Conv2d(c1, c2, 1, 1, 0, bias=False),
            nn.ELU(True)
        )

class DWConvBN(nn.Sequential):
    def __init__(self, c1, c2, k=3, s=1, p=1, d=1):
        super().__init__(
            nn.Conv2d(c1, c1, k, s, p, d, c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(True),
            nn.Conv2d(c1, c2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(True)
        )

class CPM(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.align = Conv(c1, c2, 1, 1, 0)
        self.trunk = nn.Sequential(
            DWConv(c2, c2),
            DWConv(c2, c2),
            DWConv(c2, c2)
        )
        self.conv = Conv(c2, c2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.align(x)
        identity = x
        x = self.trunk(x)
        x += identity
        x = self.conv(x)
        return x


class InitialStage(nn.Module):
    """
    nc: number of channels
    nh: number of heat maps
    np: number of pafs
    """
    def __init__(self, nc, nh, np):
        super().__init__()
        self.trunk = nn.Sequential(
            Conv(nc, nc),
            Conv(nc, nc),
            Conv(nc, nc)
        )
        self.heatmaps = nn.Sequential(
            Conv(nc, 512, 1, 1, 0),
            nn.Sequential(nn.Conv2d(512, nh, 1))
        )
        self.pafs = nn.Sequential(
            Conv(nc, 512, 1, 1, 0),
            nn.Sequential(nn.Conv2d(512, np, 1))
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.trunk(x)
        hm = self.heatmaps(x)
        pafs = self.pafs(x)
        return hm, pafs


class RefinementStageBlock(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.initial = Conv(c1, c2, 1, 1, 0)
        self.trunk = nn.Sequential(
            ConvBN(c2, c2),
            ConvBN(c2, c2, 3, 1, 2, 2)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.initial(x)
        identity = x
        x = self.trunk(x)
        x += identity
        return x


class RefinementStage(nn.Module):
    def __init__(self, c1, c2, nh, np):
        super().__init__()
        self.trunk = nn.Sequential(
            RefinementStageBlock(c1, c2),
            RefinementStageBlock(c2, c2),
            RefinementStageBlock(c2, c2),
            RefinementStageBlock(c2, c2),
            RefinementStageBlock(c2, c2)
        )
        self.heatmaps = nn.Sequential(
            Conv(c2, c2, 1, 1, 0),
            nn.Sequential(nn.Conv2d(c2, nh, 1))
        )
        self.pafs = nn.Sequential(
            Conv(c2, c2, 1, 1, 0),
            nn.Sequential(nn.Conv2d(c2, np, 1))
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.trunk(x)
        hm = self.heatmaps(x)
        pafs = self.pafs(x)
        return hm, pafs


class OpenPoseLite(nn.Module):
    def __init__(self, nh=19, np=38):
        super().__init__()
        nc = 128
        nrfs = 3
        self.model = nn.Sequential(
            ConvBN(3, 32, 3, 2, 1, bias=False),
            DWConvBN(32, 64, 3, 1, 1),
            DWConvBN(64, 128, 3, 2, 1),
            DWConvBN(128, 128, 3, 1, 1),
            DWConvBN(128, 256, 3, 2, 1),
            DWConvBN(256, 256, 3, 1, 1),
            DWConvBN(256, 512, 3, 1, 1),
            DWConvBN(512, 512, 3, 1, 2, 2),
            DWConvBN(512, 512, 3, 1, 1),
            DWConvBN(512, 512, 3, 1, 1),
            DWConvBN(512, 512, 3, 1, 1),
            DWConvBN(512, 512, 3, 1, 1),
        )
        self.cpm = CPM(512, nc)

        self.initial_stage = InitialStage(nc, nh, np)
        self.refinement_stages = nn.ModuleList([
            RefinementStage(nc+nh+np, nc, nh, np)
        for _ in range(nrfs)])

    def forward(self, x: Tensor):
        x = self.model(x)
        x = self.cpm(x)
        stages_out = []
        stages_out.extend(self.initial_stage(x))
        for refinement_stage in self.refinement_stages:
            stages_out.extend(refinement_stage(torch.cat((x, stages_out[-2], stages_out[-1]), dim=1)))
        return stages_out


if __name__ == '__main__':
    import time
    model = OpenPoseLite()
    model.load_state_dict(torch.load('checkpoints/pretrained/openposelite/openposelite_mobilenetv1_coco.pth', map_location='cpu'))
    x = torch.randn(1, 3, 192, 256)
    start = time.time()
    y = model(x)
    print((time.time()-start)*1000)
    print(y[-2].shape, y[-1].shape)
