import torch
import os.path
import numpy as np
import pandas as pd
from util import preprocess
from datasets import load_file
from datasets.basic_dataset import BasicDataset


class SingleFileDataset(BasicDataset):
    """
    A dataset class for single file paired omics dataset.
    The data should be two single files and prepared in '/path/to/data/'.
    For each single matrix file, each columns should be each sample and each row should be each molecular feature.
    """

    def __init__(self, param):
        """
        Initialize this dataset class.
        """
        BasicDataset.__init__(self, param)
        self.omics_dims = []

        # Load data for A
        A_df = load_file(param, 'A')
        # Get the min and max of A
        self.target_max = A_df.max().max()
        self.target_min = A_df.min().min()

        # Get the sample list
        if param.use_sample_list:
            sample_list_path = os.path.join(param.data_root, 'sample_list.tsv')  # get the path of sample list
            self.sample_list = np.loadtxt(sample_list_path, delimiter='\t', dtype='<U32')
        else:
            self.sample_list = A_df.columns
        # Get the feature list for A
        if param.use_feature_lists:
            feature_list_A_path = os.path.join(param.data_root, 'feature_list_A.tsv')  # get the path of feature list
            self.feature_list_A = np.loadtxt(feature_list_A_path, delimiter='\t', dtype='<U32')
        else:
            self.feature_list_A = A_df.index
        A_df = A_df.loc[self.feature_list_A, self.sample_list]
        self.A_dim = A_df.shape[0]
        self.sample_num = A_df.shape[1]
        A_array = A_df.values
        if self.param.add_channel:
            # Add one dimension for the channel
            A_array = A_array[np.newaxis, :, :]
        self.A_tensor_all = torch.Tensor(A_array)
        self.omics_dims.append(self.A_dim)

        # Load data for B
        B_df = load_file(param, 'B')
        # Get the feature list for B
        if param.use_feature_lists:
            feature_list_B_path = os.path.join(param.data_root, 'feature_list_B.tsv')  # get the path of feature list
            feature_list_B = np.loadtxt(feature_list_B_path, delimiter='\t', dtype='<U32')
        else:
            feature_list_B = B_df.index
        B_df = B_df.loc[feature_list_B, self.sample_list]
        if param.ch_separate:
            B_df_list, self.B_dim = preprocess.separate_B(B_df)
            self.B_tensor_all = []
            for i in range(0, 23):
                B_array = B_df_list[i].values
                if self.param.add_channel:
                    # Add one dimension for the channel
                    B_array = B_array[np.newaxis, :, :]
                B_tensor_part = torch.Tensor(B_array)
                self.B_tensor_all.append(B_tensor_part)
        else:
            self.B_dim = B_df.shape[0]
            B_array = B_df.values
            if self.param.add_channel:
                # Add one dimension for the channel
                B_array = B_array[np.newaxis, :, :]
            self.B_tensor_all = torch.Tensor(B_array)
        self.omics_dims.append(self.B_dim)

        if param.stratify:
            # Load labels
            labels_path = os.path.join(param.data_root, 'labels.tsv')       # get the path of the label
            labels_df = pd.read_csv(labels_path, sep='\t', header=0, index_col=0).loc[self.sample_list, :]
            self.labels_array = labels_df.iloc[:, -1].values

    def __getitem__(self, index):
        """
        Return a data point and its metadata information.

        Returns a dictionary that contains A_tensor, B_tensor
            A_tensor (tensor)           -- input data with source omics data type
            B_tensor (tensor/list)      -- output data with targeting omics data type
            index (int)                 -- the index of this data point
        """
        # Get the tensor of A
        if self.param.add_channel:
            A_tensor = self.A_tensor_all[:, :, index]
        else:
            A_tensor = self.A_tensor_all[:, index]

        # Get the tensor of B
        if self.param.ch_separate:
            B_tensor = []
            for i in range(0, 23):
                if self.param.add_channel:
                    B_tensor_part = self.B_tensor_all[i][:, :, index]
                else:
                    B_tensor_part = self.B_tensor_all[i][:, index]
                B_tensor.append(B_tensor_part)
            # Return a list of tensor
        else:
            if self.param.add_channel:
                B_tensor = self.B_tensor_all[:, :, index]
            else:
                B_tensor = self.B_tensor_all[:, index]
            # Return a tensor

        return {'A_tensor': A_tensor, 'B_tensor': B_tensor, 'index': index}

    def __len__(self):
        """
        Return the number of data points in the dataset.
        """
        return self.sample_num


