import time
import argparse
import torch
import os
import models
from util import util


class BasicParams:
    """
    This class is the father class of both TrainParams and TestParams. This class define the parameters used in both
    training and testing.
    """

    def __init__(self):
        """
        Reset the class. Indicates the class hasn't been initialized
        """
        self.initialized = False
        self.isTrain = True
        self.isTest = True

    def initialize(self, parser):
        """
        Define the common parameters used in both training and testing.
        """
        parser.add_argument('--gpu_ids', type=str, default='0',
                            help='which GPU would like to use: e.g. 0 or 0,1. -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints',
                            help='models, settings and intermediate results are saved in folder in this directory')
        parser.add_argument('--experiment_name', type=str, default='test',
                            help='name of the folder in the checkpoint directory')
        # Dataset parameters
        parser.add_argument('--data_root', required=True,
                            help='path to input data (should have sub folders train, val, test, etc)')
        parser.add_argument('--dataset_mode', type=str, default='single_file',
                            help='choose the dataset mode, options: [separate_files | single_file]')
        parser.add_argument('--batch_size', type=int, default=4,
                            help='input data batch size')
        parser.add_argument('--num_threads', default=6, type=int,
                            help='number of threads for loading data')
        parser.add_argument('--set_pin_memory', action='store_true',
                            help='set pin_memory in the dataloader to increase data loading performance')
        parser.add_argument('--file_format', type=str, default='tsv',
                            help='file format of the omics data, options: [tsv | csv | hdf]')
        parser.add_argument('--use_sample_list', action='store_true',
                            help='provide a subset sample list of the dataset, store in the path data_root/sample_list.tsv, if False use all the samples')
        parser.add_argument('--use_feature_lists', action='store_true',
                            help='provide feature lists of the input omics data, e.g. data_root/feature_list_A.tsv, if False use all the features')
        parser.add_argument('--detect_na', action='store_true',
                            help='detect missing value markers during data loading, stay False can improve the loading performance')
        parser.add_argument('--zo_norm', action='store_true',
                            help='built-in 0-1 normalisation')

        # Model parameters
        parser.add_argument('--model', type=str, default='c_gan',
                            help='chooses which model want to use, options: [c_gan | c_gan_ved]')
        parser.add_argument('--norm_type', type=str, default='batch',
                            help='the type of normalization applied to the model, default to use batch normalization, options: [batch | instance | none ]')
        parser.add_argument('--netG', type=str, default='fcg',
                            help='specify generator architecture, default is the one dimensional U-Net architecture, options: [fc_ved | fc_ved_sep | fcg | fcg_sep | fcg_single | unet_de | unet_s_de | unet_in | linear_regression]')
        parser.add_argument('--netD', type=str, default='fcd',
                            help='specify discriminator architecture, default is the one dimensional multi-layer convolution discriminator, options: [ fcd | fcd_sep | multi_conv | multi_conv_new ]')
        parser.add_argument('--input_chan_num', type=int, default=1,
                            help='number of input omics data channels, default is 1, if there are different measurements they can be different channels')
        parser.add_argument('--output_chan_num', type=int, default=1,
                            help='number of output omics channels, default is 1')
        parser.add_argument('--gen_filter_num', type=int, default=64,
                            help='number of filters in the last convolution layer in the generator')
        parser.add_argument('--dis_filter_num', type=int, default=64,
                            help='number of filters in the last convolution layer in the discriminator')
        parser.add_argument('--layer_num_D', type=int, default=3,
                            help='the number of convolution layer in the discriminator')
        parser.add_argument('--conv_k_size', type=int, default=9,
                            help='the kernel size of convolution layer, default kernel size is 9, the kernel is one dimensional.')
        parser.add_argument('--dropout_p', type=float, default=0,
                            help='probability of an element to be zeroed in a dropout layer, default is 0 which means no dropout.')
        parser.add_argument('--leaky_slope', type=float, default=0.2,
                            help='the negative slope of the Leaky ReLU activation function')
        parser.add_argument('--init_type', type=str, default='normal',
                            help='choose the method of network initialization, options: [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02,
                            help='scaling factor for normal, xavier and orthogonal initialization methods')
        parser.add_argument('--deterministic', action='store_true',
                            help='make the model deterministic for reproduction if set true')
        parser.add_argument('--seed', type=int, default=42,
                            help='random seed')
        parser.add_argument('--dist_loss', type=str, default='L1',
                            help='chooses the distance loss between the generated value and the ground truth, options: [BCE | MSE | L1]')

        # Additional parameters
        parser.add_argument('--detail', action='store_true',
                            help='print more detailed information if set true')
        parser.add_argument('--epoch_to_load', type=str, default='latest',
                            help='The epoch number to load, set latest to load latest cached model')
        parser.add_argument('--experiment_to_load', type=str, default='test',
                            help='the experiment to load')

        self.initialized = True  # set the initialized to True after we define the parameters of the project
        return parser

    def get_params(self):
        """
        Initialize our parser with basic parameters once.
        Add additional model-specific parameters.
        """
        if not self.initialized:  # check if this object has been initialized
            # if not create a new parser object
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            #  use our method to initialize the parser with the predefined arguments
            parser = self.initialize(parser)

        # get the basic parameters
        param, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = param.model
        model_param_setter = models.get_param_setter(model_name)
        parser = model_param_setter(parser, self.isTrain)

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_params(self, param):
        """
        Print welcome words and command line parameters.
        Save the command line parameters in a txt file to the disk
        """
        message = ''
        message += '\nWelcome to OmiTrans\nby Xiaoyu Zhang x.zhang18@imperial.ac.uk\n\n'
        message += '-----------------------Running Parameters-----------------------\n'
        for key, value in sorted(vars(param).items()):
            comment = ''
            default = self.parser.get_default(key)
            if value != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>18}: {:<15}{}\n'.format(str(key), str(value), comment)
        message += '----------------------------------------------------------------\n'
        print(message)

        # Save the running parameters setting in the disk
        experiment_dir = os.path.join(param.checkpoints_dir, param.experiment_name)
        util.mkdir(experiment_dir)
        file_name = os.path.join(experiment_dir, 'cmd_parameters.txt')
        with open(file_name, 'w') as param_file:
            now = time.strftime('%c')
            param_file.write('{:s}\n'.format(now))
            param_file.write(message)
            param_file.write('\n')

    def parse(self):
        """
        Parse the parameters of our project. Set up GPU device. Print the welcome words and list parameters in the console.
        """
        param = self.get_params()  # get the parameters to the object param

        param.isTrain = self.isTrain  # store this if for training or testing
        param.isTest = self.isTest

        # Print welcome words and command line parameters
        self.print_params(param)

        # Set the internal parameters
        # add_channel: add one extra dimension of channel for the input data, used for convolution layer
        # if param.netG == 'unet_1d' or param.netG == 'unet_s_1d' or param.netD == 'multi_conv_1d':
        #     param.add_channel = True
        # else:
        #     param.add_channel = False
        param.add_channel = True
        # ch_separate: separate the DNA methylation matrix base on the chromosome
        if param.netG == 'fc_ved_sep' or param.netG == 'fcg_sep':
            param.ch_separate = True
        else:
            param.ch_separate = False

        # Set up GPU
        str_gpu_ids = param.gpu_ids.split(',')
        param.gpu_ids = []
        for str_gpu_id in str_gpu_ids:
            int_gpu_id = int(str_gpu_id)
            if int_gpu_id >= 0:
                param.gpu_ids.append(int_gpu_id)
        if len(param.gpu_ids) > 0:
            torch.cuda.set_device(param.gpu_ids[0])

        self.param = param
        return self.param
