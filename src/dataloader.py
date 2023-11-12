# PyTorch imports
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import SubsetRandomSampler

# Other libraries for data manipulation and visualization
import os
import numpy as np 
import torch.utils.data as data
import os 
from src.utils_features import parallel_process
def read_data(data_dir, patient_name):
    spectrum_folder = os.path.join(data_dir, patient_name)
    loaded = np.load(os.path.join(spectrum_folder,"data.npz"), allow_pickle=True)
    print(loaded.keys())
    # if balance:
    #     non_spindle_ind = np.argwhere(loaded["labels"]!='spindle')
    #     spindle_ind = np.argwhere(loaded["labels"]=='spindle')
    #     if len(non_spindle_ind) > len(spindle_ind):
    #         sampled_ind = np.random.choice(non_spindle_ind, len(spindle_ind), replace=False)
    #         minority_ind = spindle_ind
    #     else:
    #         sampled_ind = np.random.choice(spindle_ind, len(non_spindle_ind), replace=False)
    #         minority_ind = non_spindle_ind
    #     feature = np.concatenate((loaded["feature"][sampled_ind], loaded["feature"][minority_ind]))
    #     label = np.concatenate((loaded["labels"][sampled_ind], loaded["labels"][minority_ind]))
    #     label = label.reshape(-1, 1)
    #     channel_names = np.concatenate((loaded["channel_names"][sampled_ind], loaded["channel_names"][minority_ind]))
    #     starts = np.concatenate((loaded["starts"][sampled_ind], loaded["starts"][minority_ind]))
    #     ends = np.concatenate((loaded["ends"][sampled_ind], loaded["ends"][minority_ind]))
    # else:
    feature = loaded["feature"]
    label = loaded["labels"]
    label = label.reshape(-1, 1)
    channel_names = loaded["channel_names"]
    starts = loaded["starts"]
    ends = loaded["ends"]
    start_end = np.concatenate((starts[:, None], ends[:, None]), axis=1)
    res = {"patient_name":patient_name,"feature": feature, "labels": label, "channel_names": channel_names, "start_end": start_end}
    return res

def create_list_dataset(data_dir):
    params = [{"data_dir":data_dir, "patient_name":patient_name} for patient_name in sorted(os.listdir(data_dir))]
    ret = parallel_process(params, read_data, 1, front_num=1, use_kwargs=True)
    # ret = []
    # for i, param in enumerate(params):
    #     ret[i] = read_data(**param)
    list_dataset = []
    for r in ret:
        if isinstance(r, Exception):
            print(r)
            continue
        list_dataset.append(HFODataset(r))
    print('list_dataset', len(list_dataset))
    return list_dataset

class HFODataset(Dataset):

    def __init__(self, loaded):

        self.feature = loaded["feature"]
        self.label = (loaded["labels"]=='spindle').astype(float).reshape(-1, 1)
        self.channel_names = np.squeeze(loaded["channel_names"])
        self.start_end = np.squeeze(loaded["start_end"]).reshape(-1, 2).astype(int)
        self.patient_name = loaded["patient_name"]
        self.length = len(self.feature)
        self.flip = True
       
    def __len__(self):
        
        # Return the total number of data samples
        return self.length


    def __getitem__(self, ind):
        """Returns the image and its label at the index 'ind' 
        (after applying transformations to the image, if specified).
        
        Params:
        -------
        - ind: (int) The index of the image to get

        Returns:
        --------
        - A tuple (image, label)
        """

        feature = torch.from_numpy(self.feature[ind])#[:, index+40:index+184]
        label = self.label[ind]
        channel_names = self.channel_names[ind]
        start_end = self.start_end[ind]
        
        chance = random.random()
        if self.flip and chance < 0.5:
            feature = torch.flip(feature, [2])
        return self.patient_name, feature, label, channel_names, start_end

class JITLoadDataset(Dataset):
    def __init__(self, data_dir, patient_name):
        self.data_dir = data_dir
        self.patient_name = patient_name
        self.patient_dir = os.path.join(data_dir, patient_name)
        self.files = os.listdir(self.patient_dir)
        self.length = len(self.files)
        self.flip = True
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, ind):
        loaded = np.load(os.path.join(self.data_dir, self.files[ind]), allow_pickle=True)
        feature = torch.from_numpy(loaded["feature"])
        label = loaded["labels"]
        label = label.reshape(-1, 1)
        channel_names = loaded["channel_names"]
        starts = loaded["starts"]
        ends = loaded["ends"]
        start_end = np.concatenate((starts[:, None], ends[:, None]), axis=1)
        chance = random.random()
        if self.flip and chance < 0.5:
            feature = torch.flip(feature, [2])
        return self.patient_name, feature, label, channel_names, start_end

def create_patient_eliminate_loader(data_dir, test_set_index, batch_size, seed=0, transform=transforms.ToTensor(),
                         p_val=0.2, p_test=0.2, shuffle=True, extras={}):
     
    list_of_datasets = []
    for j in sorted(os.listdir(data_dir)):
        if os.path.isdir(os.path.join(data_dir, j)):
            list_of_datasets.append(HFODataset(data_dir=data_dir, patient_name=j, transform=transform))
    # once all single json datasets are created you can concat them into a single one:
    hfo_dataset = data.ConcatDataset(list_of_datasets)
    
    testing_set = list_of_datasets[test_set_index]
    list_of_datasets.pop(test_set_index)
    
    training_set = data.ConcatDataset(list_of_datasets)
    dataset_size = len(training_set)
    all_indices = list(range(dataset_size))
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(all_indices)
    
    # Create the validation split from the full dataset
    val_split = int(np.floor(p_val * dataset_size))
    train_ind, val_ind = all_indices[val_split :], all_indices[: val_split]

    num_workers = 0
    pin_memory = False
    # If CUDA is available
    if extras:
        num_workers = extras["num_workers"]
        pin_memory = extras["pin_memory"]
    
    val_split = int(np.floor(p_val * dataset_size))
    train_ind, val_ind = all_indices[val_split :], all_indices[: val_split]
    
    sample_train = SubsetRandomSampler(train_ind)
    sample_val = SubsetRandomSampler(val_ind)
    
    train_loader = DataLoader(training_set, batch_size=batch_size, 
                            sampler=sample_train, num_workers=num_workers, 
                            pin_memory=pin_memory)
    val_loader = DataLoader(training_set, batch_size=batch_size,
                        sampler=sample_val, num_workers=num_workers, 
                            pin_memory=pin_memory)
    
    test_loader = DataLoader(testing_set, batch_size=batch_size, num_workers=num_workers, 
                            pin_memory=pin_memory)
    
    # Return the training, validation, test DataLoader objects
    return (train_loader, val_loader, test_loader)


def create_patient_loader(data_dir, patient_name, batch_size, seed=0, transform=transforms.ToTensor(),
                         p_val=0.2, p_test=0.2, shuffle=True, extras={}):
     
    testing_set = HFODataset(data_dir=data_dir, patient_name=patient_name, transform=transform, remove_artifacts= True) 

    num_workers = 8
    pin_memory = False
    # If CUDA is available
    if extras:
        num_workers = extras["num_workers"]
        pin_memory = extras["pin_memory"]
     
    test_loader = DataLoader(testing_set, batch_size=batch_size, num_workers=num_workers, 
                            pin_memory=pin_memory)
    
    # Return the training, validation, test DataLoader objects
    return (test_loader)


def create_kfold_loader(data_dir, folder_num, batch_size, seed=0, transform=transforms.ToTensor(),
                         p_val=0.1, p_test=0.2, shuffle=True, extras={}):
    # list_of_datasets = []
    # for j in sorted(os.listdir(data_dir)):
    #     list_of_datasets.append(HFODataset(data_dir=data_dir, patient_name=j, transform=transform))
    list_of_datasets = create_list_dataset(data_dir)
    # once all single json datasets are created you can concat them into a single one:
    hfo_dataset = data.ConcatDataset(list_of_datasets)
    
    dataset_size = len(hfo_dataset)
    
    all_indices = list(range(dataset_size))
    # Create the validation split from the full dataset
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(all_indices)
    test_split = int(np.floor(p_test * dataset_size))
    test_start = (folder_num - 1)*test_split
    test_end = test_start + test_split
    test_ind = all_indices[test_start: test_end]
    remain_ind = list(set(all_indices) - set(test_ind))
    fixed_val_percent = int(np.floor(p_val * dataset_size))
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(remain_ind)
    val_ind, train_ind = remain_ind[:fixed_val_percent] , remain_ind[fixed_val_percent:]

    num_workers = 1
    pin_memory = True
    # If CUDA is available
    if extras:
        num_workers = extras["num_workers"]
        pin_memory = extras["pin_memory"]

    sample_train = SubsetRandomSampler(train_ind)
    sample_val = SubsetRandomSampler(val_ind)
    sample_test = SubsetRandomSampler(test_ind)
    
    train_loader = DataLoader(hfo_dataset, batch_size=batch_size, 
                            sampler=sample_train, num_workers=num_workers, 
                            pin_memory=pin_memory)
    val_loader = DataLoader(hfo_dataset, batch_size=batch_size,
                        sampler=sample_val, num_workers=num_workers, 
                            pin_memory=pin_memory)
    
    test_loader = DataLoader(hfo_dataset,sampler=sample_test,batch_size=batch_size, num_workers=num_workers, 
                            pin_memory=pin_memory)
    
    # Return the training, validation, test DataLoader objects
    return (train_loader, val_loader, test_loader)

