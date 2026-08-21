"""
@author: ben
"""
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix
import os

# Import custom network builder
from network_builder import Net

# Disable cuDNN benchmark mode for heterogeneous Windows experiment workloads
torch.backends.cudnn.benchmark = False

### Initialization call
def init(network, init_flag):
    def he_init(network):
        if isinstance(network, (nn.Conv2d, nn.Linear)):
            ### He initialization with in-place memory
            nn.init.kaiming_uniform_(network.weight)
            ### Bias terms begin with 0s
            network.bias.data.fill_(0)
    def xavier_init(network):
        if isinstance(network, (nn.Conv2d, nn.Linear)):
            ### Xavier initialization with in-place memory
            nn.init.xavier_uniform_(network.weight)
            ### Bias terms begin with 0s
            network.bias.data.fill_(0)
    
    ### Initialize parameter sets following He or Xavier initialization
    # Handle both integer and string representations
    if init_flag == "kaiming_uniform":
        network.apply(he_init)
    else:
        network.apply(xavier_init) 
    return network  

def train_n_epochs(network: nn.Module, 
                   optimizer: torch.optim.Optimizer, 
                   train_loader: torch.utils.data.DataLoader, 
                   n_epoch: int,
                   device: torch.device,
                   base_path: str) -> tuple:
    """Train network for n epochs."""
    loss_func = nn.CrossEntropyLoss()

    # Initialize gradscaler for mixed precision training on CUDA only
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    # Initialize training dictionary
    train_dict = {'epoch': [],
                  'loss_list': []
                 }
    
    # Move network to gpu if available
    network = network.to(device)
    network.train()
    initial_loss = None

    # Train for n_epoch epochs
    for epoch in range(n_epoch):
        ### Iterate over all training data
        for batch_idx, (data, target) in enumerate(train_loader):
            
            # Push data to gpu if available
            data = data.to(device, 
                           non_blocking=True) # faster data transfer
            target = target.to(device, 
                               non_blocking=True) # faster data transfer
            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device.type):
                output = network(data)
                loss = loss_func(output, target)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        
            with torch.no_grad():
                if (batch_idx == 0) and (epoch==0):
                    initial_loss = loss.item()
                    train_dict['epoch'].append(epoch)
                    train_dict['loss_list'].append(loss.item())
            
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, 
                    batch_idx * len(data), 
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), 
                    loss.item()
                    ), flush=True)
        
        train_dict['epoch'].append(epoch)
        train_dict['loss_list'].append(loss.item())

    return network, optimizer, initial_loss, train_dict

def evaluate_network(network: nn.Module, 
                     number_of_classes: int,
                     test_loader: torch.utils.data.DataLoader, 
                     device: torch.device) -> tuple:
    """Evaluate network with vectorized metric calculations."""
    network.to(device)
    network.eval()
    
    test_loss = 0
    correct = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type):
            for data, target in test_loader:
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                output = network(data)
                
                # use cross_entropy for loss (reduction='sum' to aggregate manually)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                
                # get predictions (indices of max logit)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                
                # store all predictions and targets
                all_preds.append(pred.cpu().numpy())
                all_targets.append(target.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    # calculate metrics
    total_samples = len(test_loader.dataset)
    test_loss /= total_samples
    accuracy = 100. * correct / total_samples

    # Vectorized Metric Calculation using a Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds, labels=range(number_of_classes))
    
    TP = np.diag(cm)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    TN = cm.sum() - (FP + FN + TP)

    true_counts = cm.sum(axis=1) 
    pred_counts = cm.sum(axis=0)
    
    return TP, FP, FN, TN, true_counts, pred_counts, accuracy


def save_trained_network(network: nn.Module,
                         dataset_name: str,
                         experiment_id: int,
                         random_seed: int,
                         optimizer_name: str,
                         model_dir: str) -> None:
    """Save trained network state dict."""
    os.makedirs(model_dir, exist_ok=True)
    save_path = f"{model_dir}trained_network_{dataset_name}_{experiment_id}_{random_seed}_{optimizer_name}.pth"
    torch.save(network.state_dict(), save_path)
    print(f"Trained network saved to {save_path}", flush=True)


def run(network: nn.Module,
        device: torch.device,
        experiment_id: int,
        random_seed: int,
        dataset_name: str,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        number_of_classes: int,
        n_epoch: int
        ) -> None:
    
    # Network should already be initialized before calling run()
    network.to(device)
    # Compile network for faster training - fuses operations where possible
    network = torch.compile(network, backend='eager')

    network, optimizer, initial_loss, train_dict = train_n_epochs(network, 
                                                                  optimizer, 
                                                                  train_loader, 
                                                                  n_epoch,
                                                                  device,
                                                                  base_path=''
                                                                  )
    save_trained_network(network,
                         dataset_name=dataset_name,
                         experiment_id=experiment_id,  # This should be dynamically set based on the experiment
                         random_seed=random_seed,   # This should also be dynamically set
                         optimizer_name=optimizer.__class__.__name__,  # Get optimizer name
                         model_dir='.\\models\\'
                         )
    
    TP, FP, FN, TN, true_counts, pred_counts, accuracy = evaluate_network(network, 
                                                                          number_of_classes,
                                                                          test_loader, 
                                                                          device)
    
    return TP, FP, FN, TN, true_counts, pred_counts, accuracy, initial_loss, network, train_dict
    
if __name__ == "__main__":

    img_size = 64
    input_dim = 3
    output_dim = 10
    conv_dim_list = [32, 64]
    conv_kernel = 5
    dropout_flag = True
    bn_flag = True
    fc_dim_list = [512]
    max_pool_flag = True

    init_flag = 1
    seed_val = 42
    number_of_classes = 10
    learning_rate = 0.01
    n_epoch = 5
    
    # Test network creation
    net = Net(img_size=img_size,
              input_dim=input_dim, 
              output_dim=output_dim, 
              conv_dim_list=conv_dim_list, 
              conv_kernel=conv_kernel,
              dropout_flag=dropout_flag, 
              bn_flag=bn_flag,
              fc_dim_list=fc_dim_list, 
              max_pool_flag=max_pool_flag)
    print(net, flush=True)

    # Test with random
    sample_input = torch.randn(3200, 3, 64, 64)
    y = torch.randint(0, 10, (3200,))  # 3200 labels for 10 classes
    y_group_values = y
    print(sample_input.shape, flush=True)
    print(y.shape, flush=True)

    # Create TensorDataset and DataLoader
    dataset = TensorDataset(sample_input, y)
    train_loader = DataLoader(dataset, 
                              batch_size=32, 
                              shuffle=True,
                              drop_last=True)  # Batch size of 32
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = init(net, init_flag=init_flag)
    
    TP, FP, FN, TN, true_counts, pred_counts, accuracy, initial_loss, network, train_dict = run(network=net,
                                                                                                device=device,
                                                                                                train_loader=train_loader,
                                                                                                test_loader=train_loader,
                                                                                                optimizer=optimizer,
                                                                                                number_of_classes=number_of_classes,
                                                                                                n_epoch=n_epoch
                                                                                                )
    
    experiment_settings = {
        'experiment_id': 1,
        'img_size': img_size,
        'input_dim': input_dim,
        'output_dim': output_dim,
        'conv_dim_list': conv_dim_list,
        'conv_kernel': conv_kernel,
        'dropout_flag': dropout_flag,
        'bn_flag': bn_flag,
        'fc_dim_list': fc_dim_list,
        'max_pool_flag': max_pool_flag,
        'init_flag': init_flag,
        'seed_val': seed_val,
        'learning_rate': learning_rate,
        'n_epoch': n_epoch
    }

    evaluation_results = {
        'experiment_id': [1] * len(TP),
        'class': list(range(10)),
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'TN': TN,
        'true_counts': true_counts,
        'pred_counts': pred_counts,
        'accuracy': [accuracy] * len(TP),
        'initial_loss': [initial_loss] * len(TP)
    }

    evaluation_results = pd.DataFrame(evaluation_results)
    train_dict = pd.DataFrame(train_dict)
    experiment_settings = pd.DataFrame([experiment_settings])

# %%
