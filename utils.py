
import torch

def get_params(model):
    return {key: val.cpu().clone() for key, val in model.state_dict().items()}

def set_params(model, params):
    model.load_state_dict(params)
