data_args = {
    "image_size": 224,
    "time_window_ms": 5000,
    "freq_min_hz": 5,
    "freq_max_hz": 30,
    "resample": 100,
    "n_jobs": 16,
    "n_feature": 1,
    "new_file_per_sample": False,
    
}

augmentation_arg = {
    "random_shift_ms": 500,
    "selected_window_size_ms": 1500,
    "selected_freq_range_hz": [10, 16],
}

args = {
    'work_dir': '.',
    'data_dir': '/home/lawrence/Spindle/Pt1_2_balanced_wavelet_data_padded',   
    'res_dir': 'result/',
    'num_epochs': 30,
    'batch_size': 64,
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

