
import torch
import torch.nn.functional as F
from utils.utils import get_params, set_params
from GCN.gcn import GCN

def train_client(client_data, server_model_params, num_features, hidden_dim, num_classes, device, learning_rate, weight_decay):
    client_params = []

    for client_subset in client_data:
        client_model = GCN(num_features, hidden_dim, num_classes).to(device)
        set_params(client_model, server_model_params)

        client_model.train()
        optimizer = torch.optim.Adam(client_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        for _ in range(50):
            optimizer.zero_grad()
            out = client_model(client_subset)
            loss = F.nll_loss(out[client_subset.train_mask], client_subset.y[client_subset.train_mask])
            loss.backward()
            optimizer.step()

        client_params.append(get_params(client_model))

    return client_params
