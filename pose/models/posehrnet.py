import torch
from torch import nn, Tensor
from .backbones import HRNet


class PoseHRNet(nn.Module):
    def __init__(self, backbone: str = 'w32', num_joints: int = 17):
        super().__init__()
        self.backbone = HRNet(backbone)
        self.final_layer = nn.Conv2d(self.backbone.all_channels[0], num_joints, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def init_pretrained(self, pretrained: str = None) -> None:
        if pretrained:
            self.load_state_dict(torch.load(pretrained, map_location='cpu'), strict=False)

    def forward(self, x: Tensor) -> Tensor:
        out = self.backbone(x)
        out = self.final_layer(out)
        return out


if __name__ == '__main__':
    model = PoseHRNet('w48')
    model.load_state_dict(torch.load('checkpoints/pretrained/posehrnet_w48_256x192.pth', map_location='cpu'))
    x = torch.randn(1, 3, 256, 192)
    y = model(x)
    print(y.shape)