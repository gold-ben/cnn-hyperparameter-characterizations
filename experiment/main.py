#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 17:46:32 2021

@author: ben
"""
import os

# MUST happen before torch/numpy imports
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

### Distro modules
import argparse
import csv

import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
from multiprocessing import Pool, Lock, Manager, Process, Queue

### Local .py modules
import settings as experiment_settings_module
import network_builder as network_builder
import train_amp as train_module
import generate_paths as gen_paths
import json
import datetime

# Set PyTorch thread counts AFTER importing torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def set_seeds(random_seed):
    """Set all random seeds and CUDA settings."""
    seeds = [torch.cuda.manual_seed, 
             torch.manual_seed, 
             np.random.seed, 
             random.seed]
    for seed_func in seeds:
        seed_func(random_seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False # changed to prevent retuning for each new seed

def _init_fn(worker_id):
    # Set different random seed for each worker
    np.random.seed(torch.initial_seed() % 2**32 + worker_id)

### Modify train_loader_func to support GPU
def train_loader_func(local_data_path: str,
                      torchvision_dataset: str,
                      device: torch.device,
                      batch_size_train: int,
                      num_workers: int = 0) -> torch.utils.data.DataLoader:

    if torchvision_dataset == 'CIFAR10':
        dataset_class = torchvision.datasets.CIFAR10
        normalization_values = ((0.4914, 0.4822, 0.4465), 
                                (0.2470, 0.2435, 0.2616))
    elif torchvision_dataset == 'CIFAR100':
        dataset_class = torchvision.datasets.CIFAR100
        normalization_values = ((0.5071, 0.4867, 0.4408), 
                                (0.2675, 0.2565, 0.2761))
    elif torchvision_dataset == 'SVHN':
        dataset_class = torchvision.datasets.SVHN
        normalization_values = ((0.4377, 0.4438, 0.4728), 
                                (0.1980, 0.2010, 0.1970))
    else:
        raise ValueError(f"Unsupported dataset: {torchvision_dataset}")
    
    transform = torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(mean=normalization_values[0], 
                                                                 std=normalization_values[1])
                                                                 ])  
    # use split='test' (SVHN) otherwise use train=True
    try:
        dataset = dataset_class(local_data_path,
                                train=True,
                                download=True,
                                transform=transform)
    except TypeError:
        # fallback for datasets that expect split='test'
        dataset = dataset_class(local_data_path,
                                split='train',
                                download=True,
                                transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset = dataset,      
                                               batch_size = batch_size_train, 
                                               shuffle = True,
                                               num_workers = num_workers,
                                               worker_init_fn = _init_fn if num_workers > 0 else None,
                                               pin_memory = True if device.type == 'cuda' else False)  # Enable pin memory for GPU
    return train_loader

### Replace hard-coded test_loader_func with dataset-aware version
def test_loader_func(local_data_path: str,
                     torchvision_dataset: str,
                     device: torch.device,
                     batch_size_test: int = 1000,
                     num_workers: int = 0) -> torch.utils.data.DataLoader:

    if torchvision_dataset == 'CIFAR10':
        dataset_class = torchvision.datasets.CIFAR10
        normalization_values = ((0.4914, 0.4822, 0.4465),
                                (0.2023, 0.1994, 0.2010))
    elif torchvision_dataset == 'CIFAR100':
        dataset_class = torchvision.datasets.CIFAR100
        normalization_values = ((0.5071, 0.4867, 0.4408),
                                (0.2675, 0.2565, 0.2761))
    elif torchvision_dataset == 'SVHN':
        dataset_class = torchvision.datasets.SVHN
        normalization_values = ((0.4377, 0.4438, 0.4728),
                                (0.1980, 0.2010, 0.1970))
    else:
        raise ValueError(f"Unsupported dataset: {torchvision_dataset}")

    transform = torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(mean=normalization_values[0], 
                                                                 std=normalization_values[1])
                                                                 ])  

    # use split='test' (SVHN) otherwise use train=False
    try:
        dataset = dataset_class(local_data_path,
                                train=False,
                                download=True,
                                transform=transform)
    except TypeError:
        # fallback for datasets that expect split='test'
        dataset = dataset_class(local_data_path,
                                split='test',
                                download=True,
                                transform=transform)

    test_loader = torch.utils.data.DataLoader(dataset = dataset,
                                              batch_size=batch_size_test,
                                              shuffle=False,
                                              num_workers=num_workers,
                                              worker_init_fn=_init_fn if num_workers > 0 else None,
                                              pin_memory=True if device.type == 'cuda' else False
                                              )
    return test_loader

def optimizer_func(optimizer_name: str,
                   network: torch.nn.Module):
    # select an optimizer
    if optimizer_name == 'sgd':
        optimizer = optim.SGD(network.parameters(), 
                              lr=0.001, # default lr
                              momentum=0, # default momentum
                              dampening=0, # default dampening
                              weight_decay=0, # default weight decay
                              nesterov=False) # vanilla sgd
    elif optimizer_name == 'sgd_nesterov':
        optimizer = optim.SGD(network.parameters(), 
                              lr=0.001, # default lr
                              momentum=0.9, # default momentum for nesterov sgd
                              dampening=0, # default dampening
                              weight_decay=0, # default weight decay
                              nesterov=True) # nesterov sgd
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(network.parameters(), 
                               lr=0.001, # default lr
                               betas=(0.9, 0.999), # default betas
                               eps=1e-08, # default eps
                               weight_decay=0) # default weight decay
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(network.parameters(),
                                lr=0.001, # default lr
                                betas=(0.9, 0.999), # default betas
                                eps=1e-08, # default eps
                                weight_decay=0.01) # default weight decay
    elif optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(network.parameters(),
                                  lr=0.01) # default lr
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    return optimizer

def get_parameter_counts(network):
    total_trainable = 0
    total_cnn = 0
    total_fc = 0
    total_bn = 0

    # iterate through named_modules to identify the layer types
    # then iterate through the parameters of those specific modules
    for name, module in network.named_modules():
        # check if the module has parameters directly attached to it
        # (to avoid double-counting via the top-level container)
        module_params = sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)
        
        total_trainable += module_params
        
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            total_cnn += module_params
        elif isinstance(module, nn.Linear):
            total_fc += module_params
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            total_bn += module_params

    return total_trainable, total_cnn, total_fc, total_bn

### Modify experiment_run function
def run_experiment_i_setup(experiment_details: dict, 
                           device,
                           seed: int = 42,
                           data_loader_workers: int = 0):
    
    ### Network setup - move to GPU
    network = network_builder.Net(img_size=experiment_details['img_size_decoded'],
                                  output_dim=experiment_details['output_dim_decoded'], 
                                  input_dim=experiment_details['input_dim_decoded'], 

                                  dropout_flag=experiment_details['dropout_flag_decoded'],
                                  bn_flag=experiment_details['bn_flag_decoded'], 
                                  max_pool_flag=experiment_details['max_pool_flag_decoded'],
                                
                                  conv_dim_list=experiment_details['conv_dim_list_decoded'], 
                                  fc_dim_list=experiment_details['fc_dim_list_decoded']
                                  ).to(device)

    # Set seeds right before initialization so different init methods get different RNG states
    set_seeds(seed)
    
    # Initialize the network with the appropriate method
    init_flag_value = experiment_details['initialization_decoded']
    network = train_module.init(network, init_flag=init_flag_value)
    total_trainable, total_cnn, total_fc, total_bn = get_parameter_counts(network)
    experiment_details.update({'total_trainable_params': total_trainable,
                               'total_trainable_cnn_params': total_cnn,
                               'total_trainable_fc_params': total_fc,
                               'total_trainable_bn_params': total_bn
                               })
    print("Number of parameters in network: ", total_trainable)

    # prepare loaders (use defaults if keys missing)
    local_data_path = experiment_details.get('local_data_path', '.\\data')
    dataset_name = experiment_details.get('dataset_decoded')
    batch_size_train = experiment_details.get('batch_size_decoded')
    batch_size_test = experiment_details.get('batch_size_test', 1000)
    experiment_id = experiment_details.get('experiment_id', 'unknown')

    train_loader = train_loader_func(local_data_path,
                                     dataset_name,
                                     device,
                                     batch_size_train,
                                     num_workers=data_loader_workers)

    test_loader = test_loader_func(local_data_path,
                                   dataset_name,
                                   device,
                                   batch_size_test,
                                   num_workers=data_loader_workers)

    # Re-seed for training to ensure reproducible training dynamics
    set_seeds(seed)
    
    TP, FP, FN, TN, true_counts, pred_counts, accuracy, initial_loss, network, train_dict = train_module.run(network=network,
                                                                                                             device=device,
                                                                                                             experiment_id=experiment_id,
                                                                                                             random_seed=seed,
                                                                                                             dataset_name=dataset_name,

                                                                                                             train_loader=train_loader,
                                                                                                             test_loader=test_loader,
                                                                                                             optimizer=optimizer_func(experiment_details['optimizer'],
                                                                                                                                     network),
                                                                                                             number_of_classes=experiment_details['output_dim_decoded'],
                                                                                                             n_epoch=experiment_details['n_epoch']
                                                                                                            )
    
    train_loader = train_loader_func(local_data_path,
                                     dataset_name,
                                     device,
                                     1024) # Larger batch size for path generation to obtain stable gradients

    gaussian_path_dict = gen_paths.gaussian_path(max_steps=experiment_details.get('n_gaussian_path_steps'), 
                                                 network=network, 
                                                 train_loader=train_loader, 
                                                 initial_loss=initial_loss, 
                                                 step_size=experiment_details.get('gaussian_path_step_size'),
                                                 device=device
                                                 )
    
    experiment_details.update({
        'gaussian_path_dict': gaussian_path_dict
    })
    
    evaluation_results = {
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'TN': TN,
        'true_counts': true_counts,
        'pred_counts': pred_counts,
        'accuracy': accuracy,
        'initial_loss': initial_loss
    }

    experiment_details.update({
        'evaluation_results': evaluation_results
    })

    # added explicit cleanup to free GPU memory after each experiment - 05182026
    del network
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    print(f"Completed experiment {experiment_details['experiment_id']}.", flush=True)
    return experiment_details


def _gpu_worker_loop(device_index, task_queue, output_file, output_file_lock):
    """Persistent GPU worker process loop."""
    device = torch.device(f'cuda:{device_index}')
    torch.cuda.set_device(device)
    try:
        device_name = torch.cuda.get_device_name(device_index)
    except Exception:
        device_name = 'unknown'
    print(f"GPU worker starting on device {device_index} ({device_name})", flush=True)

    while True:
        task = task_queue.get()
        if task is None:
            break

        i, exp_dict, seed, optimizer = task
        print(f"GPU worker {device_index} starting experiment {i}, seed {seed}, optimizer {optimizer}", flush=True)

        experiment_details = exp_dict.copy()
        settings_decoded = experiment_settings_module.experiment_setup(experiment_details)
        experiment_details.update(settings_decoded)
        experiment_details['experiment_id'] = i
        experiment_details['random_seed'] = seed
        experiment_details['optimizer'] = optimizer
        experiment_details = run_experiment_i_setup(experiment_details, device, seed, data_loader_workers=4)

        with output_file_lock:
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(make_json_serializable(experiment_details)) + "\n")

        print(f"GPU worker {device_index} completed experiment {i}", flush=True)

    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print(f"GPU worker {device_index} shutting down.", flush=True)


def make_json_serializable(obj):
    """Recursively convert common non-JSON types to JSON-serializable Python types."""
    # basic types
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    # containers
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [make_json_serializable(v) for v in obj]
    # numpy
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()

def _run_single_experiment(args):
    """Worker function for multiprocessing. Must be at module level to be picklable."""
    i, exp_dict, seed, output_file, optimizer, output_file_lock = args
    experiment_details = exp_dict.copy()
    settings_decoded = experiment_settings_module.experiment_setup(experiment_details)
    experiment_details.update(settings_decoded)
    experiment_details['experiment_id'] = i
    experiment_details['random_seed'] = seed
    experiment_details['optimizer'] = optimizer
    experiment_details = run_experiment_i_setup(experiment_details, torch.device('cpu'), seed, data_loader_workers=0)
    
    with output_file_lock:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(make_json_serializable(experiment_details)) + "\n")
    return experiment_details

def experiment_runs(use_multiprocessing=False, num_processes=None, cuda_device_index=0, random_seed=42, num_seeds=5, source='gpu0', dataset_coded='CIFAR10', persistent_gpu_workers=False):
    # Set device based on multiprocessing and persistent GPU worker flag
    if use_multiprocessing and persistent_gpu_workers:
        raise ValueError("persistent_gpu_workers cannot be combined with CPU multiprocessing mode.")

    if use_multiprocessing:
        device = torch.device('cpu')
        if num_processes is None:
            num_processes = os.cpu_count()
    elif persistent_gpu_workers:
        if not torch.cuda.is_available():
            raise ValueError("persistent_gpu_workers requires CUDA devices.")
        cuda_count = torch.cuda.device_count()
        if cuda_device_index is None:
            device_indices = list(range(cuda_count))
        elif cuda_device_index < 0 or cuda_device_index >= cuda_count:
            raise ValueError(f"CUDA device index {cuda_device_index} is invalid; available devices: 0..{cuda_count-1}")
        else:
            device_indices = [cuda_device_index]
        device = torch.device(f'cuda:{device_indices[0]}')
        print(f"Persistent GPU worker mode enabled with CUDA devices: {device_indices}")
    else:
        if torch.cuda.is_available():
            cuda_count = torch.cuda.device_count()
            if cuda_device_index is None:
                device = torch.device('cuda')
            elif cuda_device_index < 0 or cuda_device_index >= cuda_count:
                raise ValueError(f"CUDA device index {cuda_device_index} is invalid; available devices: 0..{cuda_count-1}")
            else:
                device = torch.device(f'cuda:{cuda_device_index}')
        else:
            device = torch.device('cpu')

    print(f"Using device: {device}")
    if use_multiprocessing:
        print(f"Multiprocessing enabled with {num_processes} processes")
    elif persistent_gpu_workers:
        print(f"Persistent GPU workers enabled; experiments will be dispatched to CUDA devices.")
    
    # prepare output directory and file (newline-delimited JSON)
    output_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_output_path = os.path.join(output_dir, f"experiment_results_{timestamp}_{source}_{random_seed}_{num_seeds}.jsonl")

    # create/clear the file so it's ready to accept json.dumps(...) + "\n" lines later
    with open(results_output_path, "w", encoding="utf-8") as _f:
        pass

    print(f"Initialized output file: {results_output_path}")
    results_output_file = results_output_path

    presets = {
        'local_data_path': '.\\data',
        'dataset_coded': dataset_coded,
        'n_epoch': 10,
        'n_gaussian_path_steps': 500,
        'gaussian_path_step_size': 0.001,
    }

    with open('.\\experiment_list.csv', newline='') as f:
        reader = csv.DictReader(f)
        exp_list = []
        for row in reader:
            # Properly convert CSV values: try float conversion, then to int if whole number
            converted_row = {}
            for k, v in row.items():
                try:
                    float_val = float(v)
                    # Convert to int if it's a whole number
                    converted_row[k] = int(float_val) if float_val == int(float_val) else float_val
                except (ValueError, TypeError):
                    # Keep as string if it can't be converted
                    converted_row[k] = v
            exp_dict = {**presets, **converted_row}
            exp_list.append(exp_dict)
    
    if use_multiprocessing:
        with Manager() as manager:
            output_file_lock = manager.Lock()
            tasks = []
            for optimizer in ['sgd', 'sgd_nesterov', 'adam', 'adamw', 'rmsprop']:
                for seed in range(random_seed, random_seed + num_seeds):
                    print(f"Scheduling experiments for seed {seed} with optimizer {optimizer}...", flush=True)
                    for i in range(len(exp_list)):
                        tasks.append((i, exp_list[i], seed, results_output_file, optimizer, output_file_lock))

            with Pool(num_processes) as pool:
                pool.map(_run_single_experiment, tasks)
            print(f"All experiment runs completed using multiprocessing.")
    elif persistent_gpu_workers:
        output_file_lock = Lock()
        task_queue = Queue()
        workers = []

        for device_index in device_indices:
            worker = Process(target=_gpu_worker_loop,
                             args=(device_index, task_queue, results_output_file, output_file_lock))
            worker.start()
            workers.append(worker)

        for optimizer in ['sgd', 'sgd_nesterov', 'adam', 'adamw', 'rmsprop']:
            for seed in range(random_seed, random_seed + num_seeds):
                print(f"Scheduling GPU experiments for seed {seed} with optimizer {optimizer}...", flush=True)
                for i in range(len(exp_list)):
                    task_queue.put((i, exp_list[i], seed, optimizer))

        for _ in workers:
            task_queue.put(None)

        for worker in workers:
            worker.join()

        print(f"All experiment runs completed using persistent GPU workers.")
    else:
        for seed in range(random_seed, random_seed + num_seeds):
            for optimizer in ['sgd', 'sgd_nesterov', 'adam', 'adamw', 'rmsprop']:
                print(f"Starting experiments for seed {seed} with optimizer {optimizer}...")
                for i in range(len(exp_list)):
                    print(f"Starting experiment {i}...")
                    experiment_details = exp_list[i]
                    settings_decoded = experiment_settings_module.experiment_setup(experiment_details)
                    print(f"Running experiment {i} with settings: {settings_decoded}")
                    experiment_details.update(settings_decoded)
                    experiment_details['experiment_id'] = i
                    experiment_details['random_seed'] = seed
                    experiment_details['optimizer'] = optimizer

                    experiment_details = run_experiment_i_setup(experiment_details, device, seed, data_loader_workers=4)
                    
                    # Append results to the output file
                    with open(results_output_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(make_json_serializable(experiment_details)) + "\n")

                print(f"All experiment runs completed for seed {seed}.")
    
    print(f"All experiments for all seeds have been completed.")
    return experiment_details
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the CNN loss surface experiments")
    parser.add_argument('--use-multiprocessing', action='store_true', help='Use CPU multiprocessing to run experiments')
    parser.add_argument('--num-processes', type=int, default=None, help='Number of worker processes when using multiprocessing')
    parser.add_argument('--cuda-device', type=int, default=0, help='CUDA device index to use when running on GPU, e.g. 0 or 1')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda' if torch.cuda.is_available() else 'cpu', help='Prefer device mode when not using multiprocessing')
    parser.add_argument('--random-seed', type=int, default=42, help='Base random seed for experiments')
    parser.add_argument('--num-seeds', type=int, default=5, help='Number of different random seeds to run experiments with')
    parser.add_argument('--source', type=str, default='gpu0', help='Identifier for the source of the experiment runs, used in output file naming')
    parser.add_argument('--dataset-coded', type=str, default='CIFAR10', help='Dataset to use for experiments, e.g. CIFAR10, CIFAR100, SVHN')
    parser.add_argument('--persistent-gpu-workers', action='store_true', help='Use persistent CUDA worker processes for sequential GPU experiments')

    args = parser.parse_args()
    if args.use_multiprocessing and args.device == 'cuda':
        print('Warning: multiprocessing mode uses CPU only; --device cuda will be ignored.')

    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    selected_cuda_index = args.cuda_device if args.device == 'cuda' else None
    experiment_details = experiment_runs(use_multiprocessing=args.use_multiprocessing,
                                         num_processes=args.num_processes,
                                         cuda_device_index=selected_cuda_index,
                                         random_seed=args.random_seed,
                                         num_seeds=args.num_seeds,
                                         source=args.source,
                                         dataset_coded=args.dataset_coded,
                                         persistent_gpu_workers=args.persistent_gpu_workers)
