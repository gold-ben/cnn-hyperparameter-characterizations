def decode(settings_coded):
    settings_decoded = {}
    settings_decoded['local_data_path'] = settings_coded.get('local_data_path', '.\\data')
    settings_decoded['optimizer'] = settings_coded.get('optimizer')
    settings_decoded['n_epoch'] = settings_coded.get('n_epoch')
    settings_decoded['n_gaussian_path_steps'] = settings_coded.get('n_gaussian_path_steps')

    # data selection
    if settings_coded['dataset_coded'] == 'CIFAR10':
        settings_decoded['dataset_decoded'] = 'CIFAR10'
        settings_decoded['img_size_decoded'] = 32 # 32x32 images
        settings_decoded['output_dim_decoded'] = 10 # 10 classes
        settings_decoded['input_dim_decoded'] = 3 # 3 channels (RGB)
    else:
        settings_decoded['dataset_decoded'] = 'CIFAR100'
        settings_decoded['img_size_decoded'] = 32 # 32x32 images
        settings_decoded['output_dim_decoded'] = 100 # 100 classes
        settings_decoded['input_dim_decoded'] = 3 # 3 channels (RGB)
    # batch size selection
    if settings_coded['batch_size_coded'] == -1:
        settings_decoded['batch_size_decoded'] = 64
    elif settings_coded['batch_size_coded'] == 0:
        settings_decoded['batch_size_decoded'] = 128
    else:
        settings_decoded['batch_size_decoded'] = 192
    # dropout selection (boolean)
    if settings_coded['dropout_flag_coded'] == -1:
        settings_decoded['dropout_flag_decoded'] = False
    else:
        settings_decoded['dropout_flag_decoded'] = True
    # batch norm selection (boolean)
    if settings_coded['bn_flag_coded'] == -1:
        settings_decoded['bn_flag_decoded'] = False
    else:
        settings_decoded['bn_flag_decoded'] = True
    # max pooling selection (boolean)
    if settings_coded['max_pool_flag_coded'] == -1:
        settings_decoded['max_pool_flag_decoded'] = False
    else:
        settings_decoded['max_pool_flag_decoded'] = True
    # initialization selection
    if settings_coded['initialization_coded'] == -1:
        settings_decoded['initialization_decoded'] = 'xavier_uniform'
    else:
        settings_decoded['initialization_decoded'] = 'kaiming_uniform'
    # cnn width selection
    if settings_coded['cnn_width_coded'] == -1:
        cnn_width_factor = 1
    elif settings_coded['cnn_width_coded'] == 0:
        cnn_width_factor = 2
    else:
        cnn_width_factor = 3
    # convolutional layer dimensions
    if settings_coded['conv_dim_list_coded'] == -1:
        settings_decoded['conv_dim_list_decoded'] = [cnn_width_factor * l for l in [16]]
    elif settings_coded['conv_dim_list_coded'] == 0:
        settings_decoded['conv_dim_list_decoded'] = [cnn_width_factor * l for l in [16, 16]]
    else:
        settings_decoded['conv_dim_list_decoded'] = [cnn_width_factor * l for l in [16, 16, 16]]
    # fully connected layer width selection
    if settings_coded['fc_width_coded'] == -1:
        fc_width_factor = 1
    elif settings_coded['fc_width_coded'] == 0:
        fc_width_factor = 2
    else:
        fc_width_factor = 3
    # fully connected layer dimensions
    if settings_coded['fc_dim_list_coded'] == -1:
        settings_decoded['fc_dim_list_decoded'] = [fc_width_factor * l for l in [8]]
    elif settings_coded['fc_dim_list_coded'] == 0:
        settings_decoded['fc_dim_list_decoded'] = [fc_width_factor * l for l in [8, 8]] 
    else:
        settings_decoded['fc_dim_list_decoded'] = [fc_width_factor * l for l in [8, 8, 8]]

    return settings_decoded

def experiment_setup(settings_coded):
    settings_decoded = decode(settings_coded)
    return settings_decoded

if __name__ == "__main__":
    # Example coded settings
    settings_coded_example = {
        'dataset_coded': 'CIFAR10',
        'batch_size_coded': 1,
        'dropout_flag_coded': -1,
        'bn_flag_coded': -1,
        'max_pool_flag_coded': 1,
        'initialization_coded': 1,
        'cnn_width_coded': 1,
        'conv_dim_list_coded': 1,
        'fc_width_coded': 0,
        'fc_dim_list_coded': 1,
        'optimizer': 'adam',
        'n_epoch': 100,
        'n_gaussian_path_steps': 10,
    }
    
    decoded_settings = decode(settings_coded_example)
    
    settings_coded_example.update(decoded_settings)
    print(settings_coded_example)
