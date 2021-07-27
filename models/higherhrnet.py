import torch 
from torch import nn, Tensor
from backbones import HRNet
from backbones.hrnet import BasicBlock



class HigherHRNet(nn.Module):
    def __init__(self, backbone: str = 'w32', pretrained: str = None, num_joints: int = 17):
        super().__init__()
        self.backbone = HRNet(backbone)
        channel = self.backbone.pre_stage_channels[0]
        self.final_layers = nn.ModuleList([
            nn.Conv2d(channel, num_joints*2, 1, 1, 0),
            nn.Conv2d(channel, num_joints, 1, 1, 0)
        ])

        self.deconv_layers = nn.Sequential(*[
            nn.Sequential(
                nn.ConvTranspose2d(channel+(num_joints*2), channel, 4, 2, 1, 0, bias=False),
                nn.BatchNorm2d(channel),
                nn.ReLU(True)
            ),
            BasicBlock(channel, channel),
            BasicBlock(channel, channel),
            BasicBlock(channel, channel),
            BasicBlock(channel, channel)
        ])

        
    def forward(self, x: Tensor):   # 1 x 3 x H x W
        x = self.backbone(x)        # 1x 32 x H/4 x W/4
        final_outputs = []

        y = self.final_layers[0](x)
        final_outputs.append(y)

        x = torch.cat((x, y), dim=1)
        x = self.deconv_layers(x)
        y = self.final_layers[1](x)
        final_outputs.append(y)
        return final_outputs


if __name__ == '__main__':
    import time
    model = HigherHRNet('w32')
    x = torch.randn(1, 3, 224, 224)
    start = time.time()
    y = model(x)
    print((time.time() - start) * 1000)
    for g in y:
        print(g.shape)