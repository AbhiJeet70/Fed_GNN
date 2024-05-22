
import torch

def evaluate_model(server_model, pyg_data, device):
    server_model.eval()
    out = server_model(pyg_data)
    _, pred = out.max(dim=1)
    correct = float(pred[pyg_data.test_mask].eq(pyg_data.y[pyg_data.test_mask]).sum().item())
    acc = correct / pyg_data.test_mask.sum().item()
    return acc
