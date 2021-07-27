import torch
from torch import nn, Tensor


def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


class Conv(nn.Sequential):
    def __init__(self, c1, c2, k, s=1, p=0):
        super().__init__(
            nn.Conv2d(c1, c2, k, s, p, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(True)
        )


class InvertedResidualK(nn.Module):
    def __init__(self, c1, c2, first_in_stage, k=3, s=1, d=1):
        super().__init__()
        bc = c2 // 2
        p = (k - 1) // 2 * d

        self.branch1 = None

        if first_in_stage:
            self.branch1 = nn.Sequential(
                nn.Conv2d(c1, c1, k, s, p, d, c1, bias=False),
                nn.BatchNorm2d(c1),
                nn.Conv2d(c1, bc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(bc),
                nn.ReLU(True)
            )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(c1 if first_in_stage else bc, bc, 1, 1, 0, bias=False),
            nn.BatchNorm2d(bc),
            nn.ReLU(True),
            nn.Conv2d(bc, bc, k, s, p, d, bc, bias=False),
            nn.BatchNorm2d(bc),
            nn.Conv2d(bc, bc, 1, 1, 0, bias=False),
            nn.BatchNorm2d(bc),
            nn.ReLU(True)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        if self.branch1 is None:
            x1, x2 = x.chunk(2, dim=1)
            x2 = self.branch2(x2)
        else:
            x1, x2 = self.branch1(x), self.branch2(x)
        out = torch.cat([x1, x2], dim=1)
        out = channel_shuffle(out, 2)
        return out


class ShuffleNetV2K(nn.Module):
    """Based on ShuffleNetV2 where kernel_size in stages (2,3,4) is 5 instead of 3"""
    def __init__(self):
        super().__init__()
        stages_repeats = [4, 8, 4]
        stages_out_channels = [24, 348, 696, 1392]
        output_channels = stages_out_channels[0]
    
        self.input_block = nn.Sequential(*[
            Conv(3, output_channels, 3, 2, 1)
        ])

        input_channels = output_channels
        stages = []

        for repeats, output_channels in zip(stages_repeats, stages_out_channels[1:]):
            seq = [InvertedResidualK(input_channels, output_channels, True, 5, 2)]
            for _ in range(repeats-1):
                seq.append(InvertedResidualK(output_channels, output_channels, False, 5))

            stages.append(nn.Sequential(*seq))
            input_channels = output_channels

        self.conv5 = Conv(input_channels, stages_out_channels[-1], 1, 1, 0)
        self.stage2 = stages[0]
        self.stage3 = stages[1]
        self.stage4 = stages[2]
        self.stride = 16
        self.out_features = stages_out_channels[-1]
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.input_block(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        return x


if __name__ == '__main__':
    model = ShuffleNetV2K()
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)