
import torch

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
learning_rate = 0.01
weight_decay = 5e-4
hidden_dim = 512
num_clients = 3
num_epochs = 200
