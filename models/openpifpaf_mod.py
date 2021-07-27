import torch
from torch import nn, Tensor
from backbones import ShuffleNetV2K
from heads import Cif, Caf, CompositeField3
import random



class OpenPifPaf(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base_net = ShuffleNetV2K()
        self.head_nets = nn.ModuleList([
            CompositeField3(Cif, self.base_net.out_features), 
            CompositeField3(Caf, self.base_net.out_features)
        ])

    def forward(self, x: Tensor, mask=None):
        x = self.base_net(x)
        if mask is not None:
            head_outputs = tuple(hn(x).squeeze().detach().numpy() if m else None for hn, m in zip(self.head_nets, mask))
        else:
            head_outputs = tuple(hn(x).squeeze().detach().numpy() for hn in self.head_nets)

        return head_outputs


def draw_annotations(img, annotations):
    for ann in annotations:
        keypoints = ann.data
        skeleton = ann.skeleton
        keypoints = keypoints[keypoints[:, 2] > 0.0]
        xs = keypoints[:, 0]
        ys = keypoints[:, 1]

        color = (random.randint(60, 200), random.randint(0, 255), random.randint(0, 255))

        for x, y in zip(xs, ys):
            cv2.circle(img, (int(x), int(y)), 4, color, 2)

        for j1, j2, (x1, y1, _), (x2, y2, _) in ann.decoding_order:
            if (j1+1, j2+1) in skeleton or (j2+1, j1+1) in skeleton:
                cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

    return img



if __name__ == '__main__':
    from torchvision import io
    from torchvision.transforms import functional as TF
    import matplotlib.pyplot as plt
    from decoders import CifCaf
    import time
    import cv2
    model = OpenPifPaf()
    model.load_state_dict(torch.load('checkpoints/pretrained/openpifaf/shufflenetv2k16.pt', map_location='cpu'))
    model.eval()
    decoder = CifCaf(Cif, Caf)

    image = io.read_image('test.jpg').float()
    image /= 255
    image = TF.normalize(image, (0.495, 0.456, 0.406), (0.229, 0.224, 0.225))
    image = image.unsqueeze(0)

    start = time.time()
    heads = model(image)
    print((time.time() - start) * 1000)

    for h in heads:
        print(h.shape)
    # start = time.time()
    # preds = decoder(heads)
    # print((time.time() - start) * 1000)

    # im = cv2.imread('test.jpg')
    # im = draw_annotations(im, preds)
    # plt.imshow(im[:, :, ::-1])
    # plt.show()

    # pil_im = Image.open('test.jpg')
    # im = np.asarray(pil_im)
    # annotation_painter = openpifpaf.show.AnnotationPainter()
    # with openpifpaf.show.image_canvas(im) as ax:
    #     annotation_painter.annotations(ax, preds)
    #     plt.show()