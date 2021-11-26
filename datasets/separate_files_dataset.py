import os.path
from datasets.basic_dataset import BasicDataset
from datasets.data_folder import make_dataset
import numpy as np
import pandas as pd
import torch
from natsort import natsorted


class SeparateFilesDataset(BasicDataset):
    """
    A dataset class for paired omics dataset with separate files.
    The data should be prepared in a directory '/path/to/data/'.
    The SeparateFilesDataset doesn't support use_sample_list, use_feature_lists and ch_separate.
    """

    def __init__(self, param):
        """
        Initialize this dataset class.
        """
        BasicDataset.__init__(self, param)
        dir_data = os.path.join(param.data_root, 'all')
        self.data_paths = natsorted(make_dataset(dir_data))  # get omics data paths
        self.omics_dims = []
        A_array_example = np.loadtxt(self.data_paths[0])
        B_array_example = np.loadtxt(self.data_paths[1])
        self.A_dim = len(A_array_example)
        self.omics_dims.append(self.A_dim)
        self.B_dim = len(B_array_example)
        self.omics_dims.append(self.B_dim)
        # Get sample list
        sample_list_path = os.path.join(param.data_root, 'sample_list.tsv')  # get the path of sample list
        self.sample_list = np.loadtxt(sample_list_path, delimiter='\t', dtype='<U32')
        # Get the feature list for A
        feature_list_A_path = os.path.join(param.data_root, 'feature_list_A.tsv')  # get the path of feature list
        self.feature_list_A = np.loadtxt(feature_list_A_path, delimiter='\t', dtype='<U32')
        # Load labels for stratified
        if param.stratify:
            labels_path = os.path.join(param.data_root, 'labels.tsv')  # get the path of the label
            labels_df = pd.read_csv(labels_path, sep='\t', header=0, index_col=0).loc[self.sample_list, :]
            self.labels_array = labels_df.iloc[:, -1].values

    def __getitem__(self, index):
        """
        Return a data point and its metadata information.

        Returns a dictionary that contains A_tensor, B_tensor
            A_tensor (tensor)   -- input data with source omics data type
            B_tensor (tensor)   -- output data with targeting omics data type
            index (int)         -- the index of this data point
        """
        A_path = self.data_paths[index*2]
        B_path = self.data_paths[index*2+1]

        # Get the tensor of A and B
        A_array = np.loadtxt(A_path)
        if self.param.add_channel:
            # Add one dimension for the channel
            A_array_new_axis = A_array[np.newaxis, :]
            A_tensor = torch.Tensor(A_array_new_axis)
        else:
            A_tensor = torch.Tensor(A_array)

        B_array = np.loadtxt(B_path)
        if self.param.add_channel:
            B_array_new_axis = B_array[np.newaxis, :]
            B_tensor = torch.Tensor(B_array_new_axis)
        else:
            B_tensor = torch.Tensor(B_array)

        return {'A_tensor': A_tensor, 'B_tensor': B_tensor, 'index': index}

    def __len__(self):
        """
        Return the number of data points in the dataset.
        Number of data points = number of files / 2
        """
        file_num = len(self.data_paths)
        return int(file_num / 2)


