
import torch
from deeprobust.graph.data import Dataset, Dpr2Pyg

def load_data(root='/tmp/', name='cora'):
    data = Dataset(root=root, name=name)
    pyg_data = Dpr2Pyg(data)
    return pyg_data

def split_client_data(pyg_data, num_clients):
    num_nodes_per_client = pyg_data.x.size(0) // num_clients
    client_data = []

    for i in range(num_clients):
        start_idx = i * num_nodes_per_client
        end_idx = min((i + 1) * num_nodes_per_client, pyg_data.x.size(0))

        client_mask = torch.zeros(pyg_data.x.size(0), dtype=torch.bool)
        client_mask[start_idx:end_idx] = True

        edge_mask = client_mask[pyg_data.edge_index[0]] & client_mask[pyg_data.edge_index[1]]

        client_x = pyg_data.x[client_mask]
        client_edge_index = pyg_data.edge_index[:, edge_mask]
        client_y = pyg_data.y[client_mask]
        client_train_mask = pyg_data.train_mask[client_mask]
        client_val_mask = pyg_data.val_mask[client_mask]
        client_test_mask = pyg_data.test_mask[client_mask]

        client_data.append(Dpr2Pyg(Dataset(root='/tmp/', name='cora'), x=client_x, edge_index=client_edge_index, y=client_y, train_mask=client_train_mask, val_mask=client_val_mask, test_mask=client_test_mask))

    return client_data
