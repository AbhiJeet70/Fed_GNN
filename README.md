# Federated GCN 

This repository contains an implementation of federated learning using Graph Convolutional Networks (GCNs) on the Cora dataset. The project is structured into modular Python files, each handling a specific aspect of the training process.

## Repository Structure
Fed_GNN/
├── README.md
├── config
│ └── config.py
├── utils
│ └── utils.py
├── data_loader
│ └── data_loader.py
├── gcn
│ └── gcn.py
├── train
│ └── train.py
├── evaluate
│ └── evaluate.py
└── main.py

- `config.py`: Configuration settings including device setup and hyperparameters.
- `utils.py`: Utility functions for managing model parameters.
- `data_loader.py`: Functions to load and split the Cora dataset.
- `gcn.py`: Definition of the GCN model.
- `train.py`: Training logic for federated learning across multiple clients.
- `evaluate.py`: Function to evaluate the trained model.
- `main.py`: Main script to run the entire federated learning process.

## Setup

### Prerequisites

- Python 3.6 or higher
- PyTorch
- PyTorch Geometric
- Gensim 3.8.3
- DeepRobust
- GitPython

### Installation

1. Clone the repository and switch to the `main` branch:
    ```bash
    git clone https://github.com/your-username/Fed_GNN.git
    cd Fed_GNN
    git checkout main
    ```

2. Install the required dependencies:
    ```bash
    pip install torch
    pip install torch_geometric
    pip install gensim==3.8.3
    ```

3. Install the DeepRobust library:
    ```bash
    git clone https://github.com/DSE-MSU/DeepRobust.git
    cd DeepRobust
    python setup.py install
    cd ..
    ```

### Usage

1. Run the main script:
    ```bash
    python main.py
    ```

   This will execute the federated training process, where the model is trained across multiple clients and evaluated on the entire dataset.

### Notes

- Ensure that the necessary datasets are available in the specified paths or update the paths accordingly in the `data_loader.py` script.
- The training process is configured to use CUDA if available. Make sure CUDA is properly set up if you intend to use a GPU.

## How It Works

1. **Data Loading and Splitting:** The Cora dataset is loaded and split into subsets, each representing a different client in the federated learning scenario.
2. **Model Initialization:** A GCN model is initialized on the server.
3. **Client Training:** Each client trains its own model on its subset of the data, using the parameters provided by the server.
4. **Parameter Aggregation:** The server aggregates the parameters from all clients and updates the global model.
5. **Evaluation:** The global model is evaluated on the entire dataset to track the performance across training epochs.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

