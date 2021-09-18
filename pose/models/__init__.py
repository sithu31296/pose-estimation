from .posehrnet import PoseHRNet
from .simdr import SimDR


__all__ = ['PoseHRNet', 'SimDR']


def get_pose_model(model_path: str):
    if 'posehrnet' in model_path:
        model = PoseHRNet('w32' if 'w32' in model_path else 'w48')
    elif 'simdr' in model_path:
        model = SimDR('w32' if 'w32' in model_path else 'w48')
    else:
        raise NotImplementedError
    return model