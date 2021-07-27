from collections import defaultdict
import torch
import json
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import io
from torchvision import transforms as T
from typing import Union, Tuple
from xtcocotools.coco import COCO
from xtcocotools.cocoeval import COCOeval
from xtcocotools.mask import frPyObjects, decode


class COCODataset(Dataset):
    def __init__(self, root: str, split: str = 'train', img_size: Union[Tuple[int, int], int] = (256, 512), transform=None) -> None:
        super().__init__()
        assert split in ['train', 'val']
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.transform = transform
        self.split = split
        self.root = Path(root)
        self.coco = COCO(self.root / 'annotations' / f"person_keypoints_{split}2017.json")
        self.img_ids = self.coco.getImgIds()
        self.classes = ['__background__'] + [cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.num_classes = len(self.classes)
        
        self.skeletons = self.coco.cats[1]['skeleton']
        self.num_joints = len(self.coco.cats[1]['keypoints'])

        if split == 'train':
            self.img_ids = [img_id for img_id in self.img_ids if len(self.coco.getAnnIds(img_id, iscrowd=None)) > 0] 
        
        self.img_transforms = T.Compose([
            T.CenterCrop(img_size),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        

    def __len__(self) -> int:
        return len(self.img_ids)


    def __getitem__(self, index):
        img_id = self.img_ids[index]
        image_info = self.coco.loadImgs(img_id)[0]
        image = io.read_image(str(self.root / f"{self.split}2017" / image_info['file_name']))

        if self.transform is not None: image = self.transform(image)
        image = image.float()
        image /= 255
        image = self.img_transforms(image)

        if self.split == 'train':
            annotations = self.coco.loadAnns(self.coco.getAnnIds(img_id))
            masks = self.get_mask(annotations, image_info)
            annotations = list(filter(lambda x: x['iscrowd'] == 0 or x['num_keypoints'] > 0, annotations))
            skeletons = self.get_skeletons(annotations)
            return image, masks, skeletons
        else:
            return image

    def get_skeletons(self, annotations):
        # get joints for all people in an image
        # joints = [list(zip(*[annot['keypoints'][i::3] for i in range(3)])) for annot in annotations]
        # return torch.tensor(joints, dtype=torch.float32)
        joints = np.zeros((len(annotations), self.num_joints, 3), dtype=np.float32)

        for i, annot in enumerate(annotations):
            joints[i, :self.num_joints, :3] = np.array(annot['keypoints']).reshape((-1, 3))

        return joints

    def get_mask(self, annotations, img_info):
        # get ignore masks to mask out losses
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.float32)

        for annot in annotations:
            RLE = frPyObjects(annot['segmentation'], img_info['height'], img_info['width'])

            if annot['iscrowd']:
                mask += decode(RLE)
            elif annot['num_keypoints'] == 0:
                for rle in RLE:
                    mask += decode(rle)
        return mask < 0.5


    def evaluate(self, outputs, save_dir):
        save_dir = Path(save_dir)
        if not save_dir.exists(): save_dir.mkdir()

        res_file = save_dir / f"keypoints_{self.split}_results.json"

        preds, scores, image_paths = [], [], []

        for output in outputs:
            preds.append(output['preds'])
            scores.append(output['scores'])
            image_paths.append(output['image_paths'][0])

        keypoints = defaultdict(list)

        for idx, kpts in enumerate(preds):
            img_id = self.coco.loadImgs(self.img_ids[idx])[0]['file_name'].split('.')[0]

            for idx_kpt, kpt in enumerate(kpts):
                keypoints[img_id].append({
                    'keypoints': kpt[:, :3],
                    'score': scores[idx][idx_kpt],
                    'image_id': img_id
                })
        
        valid_kpts = []
        for keypoint in keypoints.values():
            valid_kpts.append(keypoint)

        self.write_coco_keypoint_results(valid_kpts, res_file)
        results = self.do_python_keypoint_eval(res_file)
        return dict(results)


    def write_coco_keypoint_results(self, pred_keypoints, res_file):
        results = []

        for img_kpts in pred_keypoints:
            if len(img_kpts) == 0: continue
            keypoints = np.array([img_kpt['keypoints'] for img_kpt in img_kpts]).reshape(-1, self.num_joints * 3)

            for img_kpt, keypoint in zip(img_kpts, keypoints):
                kpt = keypoint.reshape((self.num_joints, 3))
                x1y1 = np.amin(kpt, axis=0)
                x2y2 = np.amax(kpt, axis=0)
                
                w = x2y2[0] - x1y1[0]
                h = x2y2[1] - x1y1[1]

                results.append({
                    'image_id': img_kpt['image_id'],
                    'category_id': 1,
                    'keypoints': keypoint.tolist(),
                    'score': img_kpt['score'],
                    'bbox': [x1y1[0], x1y1[1], w, h]
                })
        
        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)


    def do_python_keypoint_eval(self, res_file):
        """Keypoint evaluation using COCOAPI"""
        sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0
        metric_names = ['AP', 'AP50', 'AP75', 'AP(M)', 'AP(L)', 'AR', 'AR50', 'AR75', 'AR(M)', 'AR(L)']

        coco_res = self.coco.loadRes(res_file)
        coco_eval = COCOeval(self.coco, coco_res, 'keypoints', sigmas=sigmas)
        coco_eval.params.useSegm = None
        coco_eval.evalaute()
        coco_eval.accumulate()
        coco_eval.summarize()

        return list(zip(metric_names, coco_eval.stats))



if __name__ == '__main__':
    dataset = COCODataset('C:\\Users\\sithu\\Documents\\Datasets\\COCO', split='train')
    dataloader = DataLoader(dataset, batch_size=2)
    
    for image, masks, skeletons in dataloader:
        print(image.shape)
        print(masks.shape)
        print(skeletons.shape)
        break
    # for image in dataloader:
    #     print(image.shape)
    #     break
