
import torch
import statistics
from config.config import device, learning_rate, weight_decay, hidden_dim, num_clients, num_epochs
from utils.utils import get_params, set_params
from data_loader.data_loader import load_data, split_client_data
from GCN.gcn import GCN
from train.train import train_client
from evaluate.evaluate import evaluate_model

# Load data
pyg_data = load_data()
print('DATA')
print(pyg_data)

num_features = pyg_data.num_node_features
num_classes = pyg_data.y.max().item() + 1  # Determine the number of classes

# Split data into client subsets
client_data = split_client_data(pyg_data, num_clients)

# Initialize the server model
server_model = GCN(num_features, hidden_dim, num_classes).to(device)
server_optimizer = torch.optim.Adam(server_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Training loop
average_accuracy = []
for epoch in range(1, num_epochs + 1):
    server_model_params = get_params(server_model)
    client_params = train_client(client_data, server_model_params, num_features, hidden_dim, num_classes, device, learning_rate, weight_decay)

    # Aggregate client parameters and update server model
    aggregated_params = {key: sum(param[key] for param in client_params) / num_clients for key in client_params[0].keys()}
    set_params(server_model, aggregated_params)

    # Evaluate the server model on the entire dataset
    acc = evaluate_model(server_model, pyg_data, device)
    print(f'Epoch {epoch}: Accuracy = {acc:.4f}')
    average_accuracy.append(acc)

# Print the average accuracy over all epochs
print(f'Average Accuracy: {statistics.mean(average_accuracy):.4f}')
