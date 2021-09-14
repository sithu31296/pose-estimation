from .loss import *


losses = {
    'jointsmse': JointsMSELoss,
    'jointsohkm': JointsOHKMMSELoss,
    'kldiscret': KLDiscretLoss,
    'nmtnorm': NMTNORMLoss,
    'nmt': NMTLoss
}

def get_loss(cfg):
    loss_fn_name = cfg['TRAIN']['LOSS']
    assert loss_fn_name in losses.keys(), f"Unavailable loss function name >> {loss_fn_name}.\nList of available loss function names: {list(loss_fn_name.keys())}"
    return losses[loss_fn_name]()