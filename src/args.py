data_args = {
    "image_size": 224,
    "time_window_ms": 3000,
    "freq_min_hz": 10,
    "freq_max_hz": 16,
    "resample": 2000,
    "n_jobs": 16,
    "n_feature": 1,
    "new_file_per_sample": True,
    
}

augmentation_arg = {
    "random_shift_ms": 45,
    "selected_window_size_ms": 200,
    "selected_freq_range_hz": [10, 290],
}

args = {
    'work_dir': '.',
    'data_dir': '/home/lawrence/Spindle/wavelet_data',   
    'res_dir': 'result/',
    'num_epochs': 30,
    'batch_size':128,
    'learning_rate': 0.0003,
    'seed': 0,
    'p_val': 0.2,
    'p_test': 0.2,
    'device': 'cuda:1',
    'augmentation_arg': augmentation_arg,
    "save_checkpoint": True,
    # balances by sampling the same number of samples from each class
    "balance": True,
    "new_file_per_sample": data_args['new_file_per_sample']
}

