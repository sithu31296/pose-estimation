import torch
from torch import nn, Tensor


class Conv(nn.Sequential):
    def __init__(self, c1, c2, k, s=1, p=0):
        super().__init__(
            nn.Conv2d(c1, c2, k, s, p, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(True)
        )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(True)
        
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes*self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)

        self.relu = nn.ReLU(True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class HRModule(nn.Module):
    def __init__(self, num_branches, num_channels, ms_output=True):
        super().__init__()
        self.num_branches = num_branches
        self.branches = nn.ModuleList([
            nn.Sequential(*[
                BasicBlock(num_channels[i], num_channels[i])
            for _ in range(4)])
        for i in range(num_branches)])

        self.fuse_layers = self._make_fuse_layers(num_branches, num_channels, ms_output)
        self.relu = nn.ReLU(True)

    def _make_fuse_layers(self, num_branches, num_channels, ms_output=True):
        fuse_layers = []

        for i in range(num_branches if ms_output else 1):
            fuse_layer = []

            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(num_channels[j], num_channels[i], 1, bias=False),
                            nn.BatchNorm2d(num_channels[i]),
                            nn.Upsample(scale_factor=2**(j-i), mode='nearest')
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j -1:
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(num_channels[j], num_channels[i], 3, 2, 1, bias=False),
                                    nn.BatchNorm2d(num_channels[i])
                                )
                            )
                        else:
                            conv3x3s.append(Conv(num_channels[j], num_channels[j], 3, 2, 1))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        
        return nn.ModuleList(fuse_layers)

    def forward(self, x: Tensor) -> Tensor:
        for i, m in enumerate(self.branches):
            x[i] = m(x[i])

        x_fuse = []

        for i, fm in enumerate(self.fuse_layers):
            y = x[0] if i == 0 else fm[0](x[0])

            for j in range(1, self.num_branches):
                y = y + x[j] if i == j else y + fm[j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


hrnet_settings = {
    "w18": [18, 36, 72, 144],
    "w32": [32, 64, 128, 256],
    "w48": [48, 96, 192, 384]
}


class HRNet(nn.Module):
    def __init__(self, backbone: str = 'w18') -> None:
        super().__init__()
        assert backbone in hrnet_settings.keys(), f"HRNet model name should be in {list(hrnet_settings.keys())}"

        # stem
        self.conv1 = nn.Conv2d(3, 64, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(True)

        self.all_channels = hrnet_settings[backbone]

        # Stage 1
        self.layer1 = self._make_layer(64, 64, 4)
        stage1_out_channel = Bottleneck.expansion * 64

        # Stage 2
        stage2_channels = self.all_channels[:2]
        self.transition1 = self._make_transition_layer([stage1_out_channel], stage2_channels)
        self.stage2 = self._make_stage(1, 2, stage2_channels)

        # # Stage 3
        stage3_channels = self.all_channels[:3]
        self.transition2 = self._make_transition_layer(stage2_channels, stage3_channels)
        self.stage3 = self._make_stage(4, 3, stage3_channels)

        # # Stage 4
        self.transition3 = self._make_transition_layer(stage3_channels, self.all_channels)
        self.stage4 = self._make_stage(3, 4, self.all_channels, ms_output=False)

    def _make_layer(self, inplanes, planes, blocks):
        downsample = None
        if inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes*Bottleneck.expansion, 1, bias=False),
                nn.BatchNorm2d(planes*Bottleneck.expansion)
            )

        layers = []
        layers.append(Bottleneck(inplanes, planes, downsample=downsample))
        inplanes = planes * Bottleneck.expansion

        for _ in range(1, blocks):
            layers.append(Bottleneck(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_transition_layer(self, c1s, c2s):
        num_branches_pre = len(c1s)
        num_branches_cur = len(c2s)
        
        transition_layers = []

        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if c1s[i] != c2s[i]:
                    transition_layers.append(Conv(c1s[i], c2s[i], 3, 1, 1))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = c1s[-1]
                    outchannels = c2s[i] if j == i-num_branches_pre else inchannels
                    conv3x3s.append(Conv(inchannels, outchannels, 3, 2, 1))
                transition_layers.append(nn.Sequential(*conv3x3s))
    
        return nn.ModuleList(transition_layers)


    def _make_stage(self, num_modules, num_branches, num_channels, ms_output=True):
        modules = []

        for i in range(num_modules):
            # multi-scale output is only used in last module
            if not ms_output and i == num_modules - 1:
                reset_ms_output = False
            else:
                reset_ms_output = True
            modules.append(HRModule(num_branches, num_channels, reset_ms_output))

        return nn.Sequential(*modules)


    def forward(self, x: Tensor) -> Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        # Stage 1
        x = self.layer1(x)

        # Stage 2
        x_list = [trans(x) if trans is not None else x for trans in self.transition1]
        y_list = self.stage2(x_list)

        # Stage 3
        x_list = [trans(y_list[-1]) if trans is not None else y_list[i] for i, trans in enumerate(self.transition2)]
        y_list = self.stage3(x_list)

        # # Stage 4
        x_list = [trans(y_list[-1]) if trans is not None else y_list[i] for i, trans in enumerate(self.transition3)]
        y_list = self.stage4(x_list)
        return y_list[0]


if __name__ == '__main__':
    model = HRNet('w32')
    model.load_state_dict(torch.load('./checkpoints/backbone/hrnet_w32.pth', map_location='cpu'), strict=False)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)