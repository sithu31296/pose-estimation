from torchvision import transforms as T


def get_transforms(cfg):
    train_transform = T.Compose(
        T.RandomSizedCrop(cfg['TRAIN']['IMAGE_SIZE']),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.1, 0.1, 0.1),
        T.AutoAugment(),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        T.RandomErasing(0.2),
    )

    val_transform = T.Compose(
        T.Resize(tuple(map(lambda x: int(x / 0.9), cfg['EVAL']['IMAGE_SIZE']))),
        T.CenterCrop(cfg['EVAL']['IMAGE_SIZE']),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    )

    return train_transform, val_transform