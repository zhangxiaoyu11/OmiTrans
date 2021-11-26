"""
This package about data loading and data preprocessing
"""
import os
import numpy as np
import pandas as pd
import importlib
from torch.utils.data import Subset
from datasets.basic_dataset import BasicDataset
from datasets.dataloader_prefetch import DataLoaderPrefetch
from sklearn.model_selection import train_test_split


def find_dataset_using_name(dataset_mode):
    """
    Get the dataset of certain mode
    """
    dataset_filename = "datasets." + dataset_mode + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    # Instantiate the dataset class
    dataset = None
    # Change the name format to corresponding class name
    target_dataset_name = dataset_mode.replace('_', '') + 'dataset'     
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BasicDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BasicDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset


def create_dataset(param):
    """
    Create a dataset given the parameters.
    """
    dataset_class = find_dataset_using_name(param.dataset_mode)
    # Get an instance of this dataset class
    dataset = dataset_class(param)
    print("Dataset [%s] was created" % type(dataset).__name__)

    return dataset


class CustomDataLoader:
    """
    Create a dataloader for certain dataset.
    """
    def __init__(self, dataset, param, shuffle=True, enable_drop_last=False):
        self.dataset = dataset
        self.param = param

        drop_last = False
        if enable_drop_last:
            if len(dataset) % param.batch_size < 3 * len(param.gpu_ids):
                drop_last = True

        # Create dataloader for this dataset
        self.dataloader = DataLoaderPrefetch(
            dataset,
            batch_size=param.batch_size,
            shuffle=shuffle,
            num_workers=int(param.num_threads),
            drop_last=drop_last,
            pin_memory=param.set_pin_memory
        )

    def __len__(self):
        """Return the number of data in the dataset"""
        return len(self.dataset)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            yield data

    def get_A_dim(self):
        """Return the dimension of first input omics data type"""
        return self.dataset.A_dim

    def get_B_dim(self):
        """Return the dimension of second input omics data type"""
        return self.dataset.B_dim

    def get_omics_dims(self):
        """Return a list of omics dimensions"""
        return self.dataset.omics_dims

    def get_sample_list(self):
        """Return the sample list of the dataset"""
        return self.dataset.sample_list

    def get_feature_list_A(self):
        """Return the feature list A of the dataset"""
        return self.dataset.feature_list_A

    def get_values_max(self):
        """Return the maximum target value of the dataset"""
        return self.dataset.target_max

    def get_values_min(self):
        """Return the minimum target value of the dataset"""
        return self.dataset.target_min


def create_single_dataloader(param, shuffle=True, enable_drop_last=False):
    """
    Create a single dataloader
    """
    param.stratify = False
    dataset = create_dataset(param)
    dataloader = CustomDataLoader(dataset, param, shuffle=shuffle, enable_drop_last=enable_drop_last)
    sample_list = dataset.sample_list

    return dataloader, sample_list


def create_separate_dataloader(param):
    """
    Create set of dataloader (train, val, test).
    """
    full_dataset = create_dataset(param)
    full_size = len(full_dataset)
    full_idx = np.arange(full_size)

    if param.stratify:
        train_idx, test_idx = train_test_split(full_idx,
                                               test_size=param.test_ratio,
                                               train_size=param.train_ratio,
                                               shuffle=True,
                                               stratify=full_dataset.labels_array)
    else:
        train_idx, test_idx = train_test_split(full_idx,
                                               test_size=param.test_ratio,
                                               train_size=param.train_ratio,
                                               shuffle=True)

    val_idx = list(set(full_idx) - set(train_idx) - set(test_idx))

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    test_dataset = Subset(full_dataset, test_idx)

    full_dataloader = CustomDataLoader(full_dataset, param)
    train_dataloader = CustomDataLoader(train_dataset, param, enable_drop_last=True)
    val_dataloader = CustomDataLoader(val_dataset, param, shuffle=False)
    test_dataloader = CustomDataLoader(test_dataset, param, shuffle=False)

    return full_dataloader, train_dataloader, val_dataloader, test_dataloader


def load_file(param, file_name):
    """
    Load data according to the format.
    """
    if param.file_format == 'tsv':
        file_path = os.path.join(param.data_root, file_name + '.tsv')
        print('Loading data from ' + file_path)
        df = pd.read_csv(file_path, sep='\t', header=0, index_col=0, na_filter=param.detect_na)
    elif param.file_format == 'csv':
        file_path = os.path.join(param.data_root, file_name + '.csv')
        print('Loading data from ' + file_path)
        df = pd.read_csv(file_path, header=0, index_col=0, na_filter=param.detect_na)
    elif param.file_format == 'hdf':
        file_path = os.path.join(param.data_root, file_name + '.h5')
        print('Loading data from ' + file_path)
        df = pd.read_hdf(file_path, header=0, index_col=0)
    else:
        raise NotImplementedError('File format %s is supported' % param.file_format)
    return df


def save_file(param, dataframe, file_name):
    """
    Save the dataframe to disk according to the format
    """
    if param.file_format == 'tsv':
        output_path = os.path.join(param.checkpoints_dir, param.experiment_name, file_name + '.tsv')
        dataframe.to_csv(output_path, sep='\t')
    elif param.file_format == 'csv':
        output_path = os.path.join(param.checkpoints_dir, param.experiment_name, file_name + '.csv')
        dataframe.to_csv(output_path)
    elif param.file_format == 'hdf':
        output_path = os.path.join(param.checkpoints_dir, param.experiment_name, file_name + '.h5')
        dataframe.to_hdf(output_path, key='df', mode='w')
    else:
        raise NotImplementedError('File format %s is supported' % param.file_format)
