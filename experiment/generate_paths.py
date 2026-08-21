#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ben
"""
### Distro modules
import torch
import numpy as np
import random
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


def set_seeds(random_seed):
    """Set all random seeds and CUDA settings."""
    seeds = [torch.cuda.manual_seed, 
             torch.manual_seed, 
             np.random.seed, 
             random.seed]
    for seed_func in seeds:
        seed_func(random_seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False


def gaussian_path(max_steps: int, 
                  network: nn.Module, 
                  train_loader: DataLoader, 
                  initial_loss: float, 
                  step_size: float,
                  device: torch.device) -> list:
    """Generate Gaussian path with stochastic batch selection."""
    # Move network to gpu if available
    network = network.to(device)
    # Initialize loss list and step counter
    gaussian_dict = {'step': [],
                     'loss_list': [],
                     'lipschitz_list': []
                    }
    step = 0
    loss_func = nn.CrossEntropyLoss()  # Define loss function
    exceeded_threshold = False

    # Take steps until initialization loss is exceeded or max steps reached
    train_iter = iter(train_loader)
    while step < max_steps and not exceeded_threshold:
        try:
            data, targets = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            data, targets = next(train_iter)

        step += 1
        data = data.to(device)
        targets = targets.to(device)

        # Zero gradients and perform forward/backward pass
        network.zero_grad()
        output = network(data)
        loss = loss_func(output, targets)
        loss.backward()

        # Check to see if initial loss is exceeded - if it is, steps have left local region
        if loss.item() > initial_loss:
            exceeded_threshold = True
            break

        # If loss has not been exceeded, take gaussian step
        else:
            with torch.no_grad():
                # Decompose model weights
                current_weights = nn.utils.parameters_to_vector(network.parameters())
                current_gradients = nn.utils.parameters_to_vector([param.grad for param in network.parameters()])

                if step > 1:
                    numerator = current_gradients - previous_gradients
                    denominator = current_weights - previous_weights
                    # Calculate Lipschitz constant
                    lipschitz_constant = torch.norm(numerator, p=2) / torch.norm(denominator, p=2)
                    gaussian_dict['lipschitz_list'].append(lipschitz_constant.item())

                # Store current weights and gradients for next iteration
                previous_weights = current_weights.detach().clone()
                previous_gradients = current_gradients.detach().clone()

                # generate and normalize global noise
                noise = torch.randn_like(current_weights)
                noise = noise / (torch.norm(noise, p=2) + 1e-8) # normalize to unit norm
                noise = noise * step_size  # scale by step size
                
                # update and re-apply to the model
                new_weights = current_weights + noise
                nn.utils.vector_to_parameters(new_weights, network.parameters())

                if step > 1 and step % 100 == 0:
                    print(f'Step {step} Gaussian Lipschitz Constant: {lipschitz_constant.item()}', flush=True)

                # Store weights and gradients
                gaussian_dict['loss_list'].append(loss.item())
                gaussian_dict['step'].append(step)

    return gaussian_dict

if __name__ == "__main__":
    
    network = nn.Sequential(
        nn.Linear(784, 256),
        nn.LeakyReLU(),
        nn.Linear(256, 128),
        nn.LeakyReLU(),
        nn.Linear(128, 10)
    )

    # Example data
    X = torch.randn(1000, 784)  # 1000 samples, 784 features
    y = torch.randint(0, 10, (1000,))  # 1000 labels for 10 classes

    # Create TensorDataset and DataLoader
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, 
                              batch_size=32, 
                              shuffle=True)  # Batch size of 32

    gaussian_path_dict = gaussian_path(max_steps=100,
                                        network=network,
                                        train_loader=train_loader,  # Replace with actual DataLoader
                                        initial_loss=100.0,
                                        step_size=0.01,
                                        device=torch.device('cuda')
                                        )
    
