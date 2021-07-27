import torch
import cv2
import time
import argparse
from pathlib import Path
from torch import Tensor
from torchvision import transforms as T
from torchvision import io
import sys
sys.path.insert(0, '.')


class VideoCapture:
    def __init__(self, file, save_dir: Path) -> None:
        self.cap = cv2.VideoCapture(file)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        w, h = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.out = cv2.VideoWriter(save_dir / 'out.avi', cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
        self.count = 0

    def next(self):
        while self.cap.isOpened():
            start = time.time()
            image = self.cap.read()[0]
            self.count += 1

            if not image:
                break
        return image


class Model:
    def __init__(self, model_path: str, device: str = 'cpu') -> None:
        self.device = torch.device(device)
        self.model = ''
        self.model.load_state_dict(torch.load(cfg['TRAINED_MODEL'], map_location='cpu'))
        self.model = self.model.to(self.device)
        self.model.eval()
        self.pose_transform = T.Compose([
            T.Resize(cfg['TEST']['IMAGE_SIZE']),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def predict(self, image: Tensor):
        image = image.float()
        image /= 255
        image = self.pose_transform(image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='checkpoints/openpifaf/shufflenetv2k.pkl')
    parser.add_argument('--file', type=str, default='test_imgs')
    args = parser.parse_args()

    file_path = Path(args.file)

    if file_path.is_file():
        image = io.read_image(str(file_path))
    else:
        files = file_path.glob("*.jpg")
        for file in files:
            image = io.read_image(str(file))